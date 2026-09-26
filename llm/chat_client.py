"""OpenAI-compatible client for a locally running LM Studio server.

LM Studio doesn't check the API key, but the OpenAI client library
requires a non-empty string, hence the placeholder below.
"""

import json
import re

import requests
from openai import OpenAI

MAX_TOOL_ITERATIONS = 5  # guards against a runaway tool-call loop


def _fetch_context_length(base_url: str, model: str) -> int:
    """Best-effort: asks LM Studio's own extended REST API (not part of
    the OpenAI-compatible surface) for the currently loaded context
    length of `model`, e.g. http://localhost:1234/api/v0/models ->
    {"data": [{"id": "...", "loaded_context_length": 262144, ...}]}.

    Returns None if the server doesn't expose this (a different
    OpenAI-compatible backend, or an older LM Studio) -- callers must
    treat that as "unknown, don't try to compact" rather than guessing
    a number, since a wrong guess could compact either too early or
    too late.
    """
    try:
        root = base_url.rsplit("/v1", 1)[0]
        resp = requests.get(f"{root}/api/v0/models", timeout=5)
        resp.raise_for_status()
        for entry in resp.json().get("data", []):
            if entry.get("id") == model:
                return entry.get("loaded_context_length") or entry.get("max_context_length")
    except (requests.RequestException, ValueError, KeyError):
        pass
    return None


# Confirmed live: when this model is pushed to use a tool that isn't
# actually registered with the API (tools=None/[] on this request), it
# doesn't just refuse or guess -- it can hallucinate its own tool-call
# syntax as plain text content, e.g.
#   <|tool_call>call:brave_search:search{queries:["..."]}<tool_call|>
# Since message.tool_calls is empty in that case, this text falls
# through as a normal reply and would otherwise get spoken verbatim.
# The real fix is making sure tools are actually registered whenever the
# system prompt says they're available (see main.py) -- this is
# defense-in-depth for whatever slips through anyway (a model hiccup, a
# future prompt/config mismatch), same rationale as tts/tts_client.py's
# markdown stripping.
_LEAKED_TOOL_CALL = re.compile(r"<\|?/?\s*(tool_call|tool_code|function_call)\s*\|?/?>", re.IGNORECASE)
FALLBACK_REPLY = "Sorry, I had trouble with that -- could you try asking again?"


def _looks_like_leaked_tool_call(text: str) -> bool:
    return bool(text) and bool(_LEAKED_TOOL_CALL.search(text))


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        tools: list = None,
        tool_dispatch=None,
        context_limit: int = None,
        compaction_threshold: float = 0.8,
    ):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        self.temperature = temperature
        self.tools = tools
        self.tool_dispatch = tool_dispatch
        self.system_prompt = system_prompt
        self.compaction_threshold = compaction_threshold
        self.history = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

        # context_limit=None (the default) means "query LM Studio for it";
        # pass an explicit int to override, or to skip the query for a
        # backend that doesn't support it. Either way, None here means
        # needs_compaction() can never fire -- no size to compare against.
        self.context_limit = context_limit if context_limit is not None else _fetch_context_length(base_url, model)
        self.last_total_tokens = None

    def _record_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_total_tokens = usage.total_tokens

    def needs_compaction(self) -> bool:
        if self.context_limit is None or self.last_total_tokens is None:
            return False
        return self.last_total_tokens >= self.compaction_threshold * self.context_limit

    def compact(self):
        """Replaces the turn-by-turn history with a single condensed
        summary, freeing up context. Called by main.py (after speaking a
        brief notice) once needs_compaction() says it's time.

        Keeps the system prompt as-is; asks the model itself to
        summarize everything else (skipping raw tool-call plumbing,
        which is noisy and not useful once compacted) into a short note,
        then seeds history with that note as if the user and assistant
        had already covered it -- so the model has continuity without
        carrying every prior message verbatim.
        """
        conversation = [m for m in self.history if m["role"] != "system"]
        if not conversation:
            return

        transcript_lines = []
        for m in conversation:
            role = m["role"]
            if role == "tool":
                continue  # raw scraped page text etc. -- noisy, not needed in a summary
            content = m.get("content") or ""
            if not content:
                continue
            transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)
        if not transcript:
            return

        summary_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the following conversation concisely, in "
                        "third person, keeping any facts, names, "
                        "preferences, or open threads that would matter for "
                        "continuing it naturally. This will replace the raw "
                        "conversation in an assistant's context, not be "
                        "shown to anyone directly."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
            temperature=0.3,
        )
        summary = summary_response.choices[0].message.content

        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})
        self.history.append(
            {"role": "user", "content": f"(For context, here is a summary of our conversation so far: {summary})"}
        )
        self.history.append({"role": "assistant", "content": "Got it, I'll keep that in mind."})
        self.last_total_tokens = None

    def ask(self, user_text: str) -> str:
        """Blocking, non-streaming call. Returns the final reply and updates
        history, running any tool calls the model makes along the way."""
        self.history.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=self.temperature,
                tools=self.tools if self.tools else None,
            )
            message = response.choices[0].message
            self._record_usage(response)

            if not message.tool_calls:
                reply = message.content
                if _looks_like_leaked_tool_call(reply):
                    print(f"[ChatClient] discarding hallucinated tool-call text: {reply!r}")
                    reply = FALLBACK_REPLY
                self.history.append({"role": "assistant", "content": reply})
                return reply

            self.history.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls],
                }
            )
            for tc in message.tool_calls:
                arguments = json.loads(tc.function.arguments)
                result = self.tool_dispatch(tc.function.name, arguments)
                self.history.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        return "Sorry, that took too many steps to answer — could you try rephrasing it?"

    def ask_streaming(self, user_text: str):
        """Generator yielding text deltas as they arrive.

        Not wired into main.py yet (that's the phase-7 latency work —
        sentence-chunking this output into TTS as it arrives rather than
        waiting for the full reply) but the client-side piece is ready.
        """
        self.history.append({"role": "user", "content": user_text})
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=self.temperature,
            stream=True,
        )
        collected = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                collected.append(delta)
                yield delta
        self.history.append({"role": "assistant", "content": "".join(collected)})
