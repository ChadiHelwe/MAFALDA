import numpy as np

from src.classification_models.llama_based_models import BaseModel
from src.evaluate import LEVEL_2_NUMERIC


class SilentModel(BaseModel):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.instruction_begin = ""
        self.instruction_end = ""

    def predict(self, text: str, max_tokens: int = 2048, echo: bool = True):
        return "not"


class RandomModel(BaseModel):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.instruction_begin = ""
        self.instruction_end = ""
        self.predict_probability = 0.333

        labels_dict = LEVEL_2_NUMERIC.copy()
        labels_dict.pop("nothing")
        self.labels = list(labels_dict.keys())

    def predict(self, text: str, max_tokens: int = 2048, echo: bool = True):
        if np.random.rand() <= self.predict_probability:
            return np.random.choice(self.labels)
        else:
            return ""
