from openai import OpenAI

from src.classification_models.llama_based_models import BaseModel


class ChatGPTModel(BaseModel):
    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.instruction_begin = ""
        self.instruction_end = ""

    def predict(self, text: str, max_tokens: int = 2048, echo: bool = True):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
            max_tokens=max_tokens,
            temperature=0,
        )

        if echo:
            print("text:", text)
            print("GPT answer:", completion.choices[0].message.content)

        return completion.choices[0].message.content
