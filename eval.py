import logging
from dataclasses import dataclass, field

import evaluate
import numpy as np
from datasets import load_dataset
from evaluate import Text2TextGenerationEvaluator
from strenum import StrEnum
from transformers import (
    HfArgumentParser,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)

logger = logging.getLogger(__name__)


class EvaluationModule(StrEnum):
    BERTSCORE = "bertscore"


@dataclass
class EvaluateModelArguments:
    pretrained_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    max_length: int = field(
        default=32,
        metadata={"help": "The maximum total input sequence length after tokenization"},
    )
    num_beams: int = field(
        default=4,
        metadata={"help": "Number of beams to use for evaluation"},
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})


@dataclass
class EvaluateMetricArguments:
    evaluation_module: EvaluationModule = field(
        default=EvaluationModule.BERTSCORE,
        metadata={"help": "Name of the evaluation module from the HuggingFace hub"},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the results to the HuggingFace hub"},
    )


def contain_question_mark_token(data):
    return data["question_tokens"]["tokens"][-1] == "?"


def normalise(data):
    # Lowercase the text
    data["source"] = data["source"].lower()
    data["target"] = data["target"].lower()

    # Remove new line characters
    data["source"] = data["source"].replace("\n", " ")

    return data


def main():
    parser = HfArgumentParser((EvaluateModelArguments, EvaluateMetricArguments))
    model_args, metric_args = parser.parse_args_into_dataclasses()

    set_seed(model_args.seed)

    logger.info("loading dataset")

    eval_dataset = (
        load_dataset("mrqa", split="test")
        .filter(contain_question_mark_token)
        .select_columns(["context", "question"])
        .rename_columns({"context": "source", "question": "target"})
        .map(normalise)
    )

    logger.info("loading model")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.pretrained_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)

    logger.info("loading metric")
    metric = evaluate.load(metric_args.evaluation_module)

    logger.info("evaluating model")
    evaluator = Text2TextGenerationEvaluator()
    if metric_args.evaluation_module == EvaluationModule.BERTSCORE:
        evaluator.METRIC_KWARGS = {"model_type": "microsoft/deberta-xlarge-mnli"}
    results = evaluator.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=eval_dataset,
        metric=metric,
        input_column="source",
        label_column="target",
        random_state=model_args.seed,
        generation_kwargs={
            "max_new_tokens": model_args.max_length,
            "num_beams": model_args.num_beams,
        },
    )

    logger.info("saving results")
    params = {
        "evaluation_module": metric_args.evaluation_module,
        "model": model_args.pretrained_model_name_or_path,
    }
    evaluate.save("./results/", **results, **params)

    metric_value = None
    if metric_args.evaluation_module == EvaluationModule.BERTSCORE:
        mean_f1 = np.mean(results["f1"])
        metric_value = mean_f1
        mean_recall = np.mean(results["recall"])
        mean_precision = np.mean(results["precision"])
        logger.info(f"mean f1: {mean_f1}")
        logger.info(f"mean recall: {mean_recall}")
        logger.info(f"mean precision: {mean_precision}")

    if metric_args.push_to_hub and metric_value:
        logger.info("pushing results to the hub")
        evaluate.push_to_hub(
            model_id=model_args.pretrained_model_name_or_path,
            task_type="text2text-generation",
            task_name="Question Generation",
            dataset_type="mrqa",
            dataset_split="test",
            dataset_name="MRQA 2019",
            metric_value=metric_value,
            metric_type=metric_args.evaluation_module,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate.logging.set_verbosity_info()
    main()
