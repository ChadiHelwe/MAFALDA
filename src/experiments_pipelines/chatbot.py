from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic.v1 import validator


class ChatBotLLM(LLM):
    max_length: int = 4096
    model: Any

    @validator("max_length")
    def validate_max_length(cls, v):
        if v < 0:
            raise ValueError("max_length must be non-negative")
        return v

    @validator("model")
    def validate_model(cls, v):
        if not hasattr(v, "predict"):
            raise ValueError("model must have a predict method")
        return v

    @property
    def _llm_type(self) -> str:
        return self.model.model_name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        prediction = self.model.predict(prompt, self.max_length)
        return prediction

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_length": self.max_length, "model": self.model.model_name}
