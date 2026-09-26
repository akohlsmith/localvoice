"""OpenAI-compatible client for a locally running LM Studio server.

LM Studio doesn't check the API key, but the OpenAI client library
requires a non-empty string, hence the placeholder below.
"""

from openai import OpenAI


class ChatClient:
    def __init__(self, base_url: str, model: str, system_prompt: str = "", temperature: float = 0.7):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model
        self.temperature = temperature
        self.history = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def ask(self, user_text: str) -> str:
        """Blocking, non-streaming call. Returns the reply and updates history."""
        self.history.append({"role": "user", "content": user_text})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=self.temperature,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

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
