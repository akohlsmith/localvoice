"""OpenAI-compatible client for a locally running LM Studio server.

LM Studio doesn't check the API key, but the OpenAI client library
requires a non-empty string, hence the placeholder below.
"""

import json

from openai import OpenAI

MAX_TOOL_ITERATIONS = 5  # guards against a runaway tool-call loop


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        tools: list = None,
        tool_dispatch=None,
    ):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        self.temperature = temperature
        self.tools = tools
        self.tool_dispatch = tool_dispatch
        self.history = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

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

            if not message.tool_calls:
                reply = message.content
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
