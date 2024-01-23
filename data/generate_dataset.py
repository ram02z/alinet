import logging
import os
from dataclasses import dataclass, field

from datasets import load_dataset, concatenate_datasets
from transformers import set_seed, HfArgumentParser
from strenum import StrEnum

logger = logging.getLogger(__name__)


class Dataset(StrEnum):
    BASELINE_TRAIN = "baseline_train"
    RC_EVAL = "reading_comprehension_eval"

@dataclass
class GenerateDatasetArguments:
    dataset: Dataset = field(metadata={"help": "Name of the dataset"})
    data_dir: str = field(default="data", metadata={"help": "Output data directory"})
    seed: int = field(default=42, metadata={"help": "Random seed"})

def contain_question_mark(data):
    return data["target"][-1].rstrip() == "?"


def contain_unique_question_context(data, unique_sources):
    if data["source"] not in unique_sources:
        unique_sources.add(data["source"])
        return True
    return False


def normalise(data):
    # Lowercase the text
    data["source"] = data["source"].lower()
    data["target"] = data["target"].lower()

    # Remove new line characters
    data["source"] = data["source"].replace("\n", " ")

    return data


def main():
    parser = HfArgumentParser((GenerateDatasetArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    logger.info("loading datasets")
    if args.dataset == Dataset.BASELINE_TRAIN:
        data = (
            load_dataset("squad")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )
    elif args.dataset == Dataset.RC_EVAL:
        # BUG: https://huggingface.co/datasets/mrqa/discussions/3
        data = (
            load_dataset("mrqa", split="test")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )

    logger.info("saving dataset")

    if args.dataset == Dataset.BASELINE_TRAIN:
        data["train"].to_csv(os.path.join(args.data_dir, "train.csv"))
        data["validation"].to_csv(os.path.join(args.data_dir, "validation.csv"))
    elif args.dataset == Dataset.RC_EVAL:
        data.to_csv(os.path.join(args.data_dir, "test.csv"))


if __name__ == "__main__":
    main()
