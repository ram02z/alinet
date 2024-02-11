import logging
from dataclasses import dataclass, field
import unicodedata

import evaluate
import numpy as np
import datasets
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


class Dataset(StrEnum):
    RC = "reading_comprehension"
    NOISE = "spoken_noise"


DATASETS = {
    Dataset.RC: {"id": "mrqa", "name": "MRQA"},
    Dataset.NOISE: {"id": "alinet/spoken_squad", "name": "Spoken-SQuAD"},
}

METRICS = {EvaluationModule.BERTSCORE: {"id": "bertscore", "name": "BERTScore"}}


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


@dataclass
class EvaluateDataArguments:
    dataset: Dataset = field(metadata={"help": "Name of the dataset to use"})


def contain_question_mark(data):
    return data["target"][-1].rstrip() == "?"


def normalise(data):
    # Remove new line characters
    data["source"] = data["source"].replace("\n", " ")

    # Resolve accented characters
    data["source"] = "".join(
        c
        for c in unicodedata.normalize("NFD", data["source"])
        if unicodedata.category(c) != "Mn"
    )
    data["target"] = "".join(
        c
        for c in unicodedata.normalize("NFD", data["target"])
        if unicodedata.category(c) != "Mn"
    )

    return data


def main():
    parser = HfArgumentParser(
        (EvaluateModelArguments, EvaluateMetricArguments, EvaluateDataArguments)
    )
    model_args, metric_args, data_args = parser.parse_args_into_dataclasses()

    set_seed(model_args.seed)

    logger.info("loading dataset")

    if data_args.dataset == Dataset.RC:
        eval_data = (
            load_dataset("mrqa", split="test")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )
    elif data_args.dataset == Dataset.NOISE:
        eval_data = (
            load_dataset("alinet/spoken_squad", name="WER54", split="test")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )

    logger.info("loading model")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.pretrained_model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)

    logger.info("loading metric")
    metric_meta = METRICS.get(metric_args.evaluation_module)
    if not metric_meta:
        raise RuntimeError("could not get metric metadata")
    metric = evaluate.load(metric_meta["id"])

    logger.info("evaluating model")
    evaluator = Text2TextGenerationEvaluator()
    if metric_args.evaluation_module == EvaluationModule.BERTSCORE:
        evaluator.METRIC_KWARGS = {"model_type": "microsoft/deberta-xlarge-mnli"}
    results = evaluator.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=eval_data,
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
        "dataset": data_args.dataset,
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

    dataset_meta = DATASETS.get(data_args.dataset)

    if metric_args.push_to_hub and metric_value and dataset_meta:
        logger.info("pushing results to the hub")
        evaluate.push_to_hub(
            model_id=model_args.pretrained_model_name_or_path,
            task_type="text2text-generation",
            task_name="Question Generation",
            dataset_type=dataset_meta["id"],
            dataset_name=dataset_meta["name"],
            metric_value=metric_value,
            metric_name=metric_meta["name"],
            metric_type=metric_meta["id"],
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    evaluate.logging.set_verbosity_info()
    main()
