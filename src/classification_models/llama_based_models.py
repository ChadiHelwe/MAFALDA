import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from transformers.models.llama.convert_llama_weights_to_hf import write_model


class BaseModel:
    def predict(self, x, max_length: int = 4096, device: str = "cpu"):
        with torch.no_grad():
            if self.device is None:
                self.device = device
                self.model.to(device)

            tokenized = self.tokenizer(x, return_tensors="pt")
            generate_ids = self.model.generate(
                tokenized.input_ids.to(device), max_length=max_length
            )
            output = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output


class LLaMAModel(BaseModel):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_size: str = "7B",
    ) -> None:
        if not os.path.exists(output_dir):
            write_model(
                output_dir, os.path.join(input_dir, model_size), model_size=model_size
            )
        self.model_name = "LLaMA"
        self.device = None
        self.tokenizer = LlamaTokenizer.from_pretrained(f"{input_dir}/tokenizer.model")
        self.model = LlamaForCausalLM.from_pretrained(output_dir)
        self.model.eval()


class LLaMAModelQuantized(BaseModel):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_size: str = "7B",
    ) -> None:
        if not os.path.exists(output_dir):
            write_model(
                output_dir, os.path.join(input_dir, model_size), model_size=model_size
            )
        self.model_name = "LLaMA"
        self.device = None
        self.tokenizer = LlamaTokenizer.from_pretrained(f"{input_dir}/tokenizer.model")
        self.model = LlamaForCausalLM.from_pretrained(output_dir, load_in_8bit=True)
        self.model.eval()

    def predict(self, x, max_length: int = 4096):
        with torch.no_grad():
            tokenized = self.tokenizer(x, return_tensors="pt")
            generate_ids = self.model.generate(
                tokenized.input_ids.to("cuda:0"), max_length=max_length
            )
            output = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output


class VicunaModel(BaseModel):
    def __init__(self, model_size: str = "7B") -> None:
        self.model_name = "Vicuna"
        self.device = None
        if model_size == "7B":
            self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
            self.model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3")
            self.model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.3")
        self.model.eval()


class AlpacaModel(BaseModel):
    def __init__(self, model_size: str = "7B") -> None:
        self.model_name = "Alpaca"
        self.device = None
        if model_size == "7B":
            self.tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native")
            self.model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-13b")
            self.model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-13b")
        self.model.eval()


class MPTModel(BaseModel):
    def __init__(self, model_size: str = "7B") -> None:
        self.model_name = "MPT"
        self.device = None
        if model_size == "7B":
            mpt_model = "mosaicml/mpt-7b"
        else:
            mpt_model = "mosaicml/mpt-13b"

        config = AutoConfig.from_pretrained(mpt_model, trust_remote_code=True)
        config.max_seq_len = 4096
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.model = AutoModelForCausalLM.from_pretrained(
            mpt_model, trust_remote_code=True
        )
        self.model.eval()


class DollyModel(BaseModel):
    def __init__(self, model_size: str = "7B") -> None:
        self.model_name = "Dolly"
        self.device = None
        if model_size == "7B":
            self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b")
            self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b")
        elif model_size == "6B":
            self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b")
            self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b")
        elif model_size == "3B":
            self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
            self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
            self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b")
        self.model.eval()


if __name__ == "__main__":
    model = LLaMAModel("LLaMA/", "Converted_Llama/", model_size="7B")
