import argparse

from src.classification_models.quantized_llama_based_models import (
    LLaMABasedQuantizedModel,
)
from src.classification_models.openai_based_models import ChatGPTModel
from src.classification_models.baseline_models import RandomModel, SilentModel

from src.evaluate import eval_dataset
from src.experiments_pipelines.pipelines import zero_or_few_shots_pipeline
from src.utils import setup_logger


def run_experiment(
    model: str,
    size: str,
    quantization: str,
    level: int,
    n_gpu_layers: int = 0,
):
    model = LLaMABasedQuantizedModel(
        model_size=size,
        model_name=model,
        quantization=quantization,
        n_gpu_layers=n_gpu_layers,
    )

    zero_or_few_shots_pipeline(
        model=model,
        dataset_path="datasets/gold_standard_dataset.jsonl",
        prediction_path=f"results/{model.model_name}_{size}_{quantization}_level_{level}_results.jsonl",
        level=level,
    )

def run_chatgpt_experiment(
        model_name: str,
        level: int,
):
    model = ChatGPTModel(model_name=model_name)
    zero_or_few_shots_pipeline(
        model=model,
        dataset_path="datasets/gold_standard_dataset.jsonl",
        prediction_path=f"results/{model.model_name}_level_{level}_results.jsonl",
        level=level,
    )

def run_baseline_experiment(
        model_name: str,
        level: int,
):
    if model_name == "base-silent":
        model = SilentModel(model_name=model_name)
    elif model_name == "base-random":
        model = RandomModel(model_name=model_name)
    else:
        raise Exception("You must select a valid baseline model, e.g. base-random")

    zero_or_few_shots_pipeline(
        model=model,
        dataset_path="datasets/gold_standard_dataset.jsonl",
        prediction_path=f"results/{model.model_name}_level_{level}_results.jsonl",
        level=level,
    )

def evaluate():
    eval_dataset("datasets/gold_standard_dataset.jsonl", "results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--size", type=str, help="Model size")
    parser.add_argument("--quantization", type=str, help="Quantization")
    parser.add_argument("--level", type=int, help="Level")
    parser.add_argument(
        "--n_gpu_layers", type=int, help="Number of GPUs for layer", default=0
    )
    parser.add_argument("--eval", help="Evaluate", action="store_true")

    args = parser.parse_args()

    logger_filename = (
        f"logs/{args.model}_{args.size}_{args.quantization}_level_{args.level}.log"
    )
    logger = setup_logger(logger_filename)
    try:
        if args.eval:
            evaluate()
        else:
            if args.model[:3] == "gpt":
                run_chatgpt_experiment(
                    model_name=args.model,
                    level=args.level,
                )
            elif args.model[:4] == "base":
                run_baseline_experiment(
                    model_name=args.model,
                    level=args.level,
                )
            else:
                run_experiment(
                    model=args.model,
                    size=args.size,
                    quantization=args.quantization,
                    level=args.level,
                    n_gpu_layers=args.n_gpu_layers,
                )

    except Exception as e:
        logger.error(e, exc_info=True)