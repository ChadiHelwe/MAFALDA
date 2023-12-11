import numpy as np

from src.evaluate import LEVEL_2_NUMERIC


class SilentModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self):
        return "not"


class RandomModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.predict_probability = 0.333

        labels_dict = LEVEL_2_NUMERIC.copy()
        labels_dict.pop("nothing")
        self.labels = list(labels_dict.keys())

    def predict(self):
        if np.random.rand() <= self.predict_probability:
            return np.random.choice(self.labels)
        else:
            return ""
