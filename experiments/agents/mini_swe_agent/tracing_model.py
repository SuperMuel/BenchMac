from __future__ import annotations

from typing import Any

from minisweagent.models.litellm_model import LitellmModel


class TracingModel(LitellmModel):
    """A LitellmModel subclass that records raw LiteLLM responses.

    Keeps the same public API and attributes so it remains compatible with
    minisweagent's DefaultAgent while adding response collection.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.responses: list[dict[str, Any]] = []

    def query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict:
        out: dict = super().query(messages, **kwargs)
        try:
            raw = out.get("extra", {}).get("response")
            if raw is not None:
                self.responses.append(raw)
        except Exception:
            # Tracing must not interfere with agent execution
            pass
        return out
