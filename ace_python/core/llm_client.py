"""
ACE - LLM Client
Supports Anthropic Claude with streaming, JSON mode, and retry logic.
"""
import json
import os
import time
from typing import Iterator, Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class LLMClient:
    """
    Unified LLM client supporting streaming and JSON-structured outputs.
    Uses Anthropic SDK directly or via LangChain.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-5",
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Set it via environment variable or pass api_key parameter."
            )

    def _get_client(self):
        if self._client is None:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        max_retries: int = 3,
    ) -> str:
        """Single completion call."""
        client = self._get_client()
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = client.messages.create(**kwargs)
                return response.content[0].text
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def stream(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> Iterator[str]:
        """Streaming completion. Yields text chunks."""
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def complete_json(
        self,
        prompt: str,
        system_prompt: str = "",
        schema_hint: str = "",
        max_retries: int = 3,
    ) -> Optional[dict]:
        """Returns parsed JSON dict or None on failure."""
        json_system = (
            system_prompt
            + "\n\nCRITICAL: Respond ONLY with valid JSON. "
            "No markdown fences, no preamble, no explanation."
        )
        if schema_hint:
            json_system += f"\n\nExpected schema:\n{schema_hint}"

        for attempt in range(max_retries):
            try:
                text = self.complete(prompt, json_system)
                # Strip markdown fences if present
                clean = text.strip()
                if clean.startswith("```"):
                    clean = "\n".join(clean.split("\n")[1:])
                if clean.endswith("```"):
                    clean = "\n".join(clean.split("\n")[:-1])
                return json.loads(clean.strip())
            except json.JSONDecodeError:
                if attempt < max_retries - 1:
                    continue
                return None

    def langchain_chat(self, messages: list) -> str:
        """Use via LangChain if available."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-anthropic not installed")
        llm = ChatAnthropic(model=self.model, api_key=self.api_key)
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))
        result = llm.invoke(lc_messages)
        return result.content
