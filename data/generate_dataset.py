import logging
import os
from dataclasses import dataclass, field

import datasets
from datasets import load_dataset
from transformers import set_seed, HfArgumentParser
from strenum import StrEnum

logger = logging.getLogger(__name__)


class Dataset(StrEnum):
    BASELINE_TRAIN = "baseline_train"

@dataclass
class GenerateDatasetArguments:
    dataset: Dataset = field(metadata={"help": "Name of the dataset to use"})
    data_dir: str = field(default="data", metadata={"help": "Output data directory"})
    seed: int = field(default=42, metadata={"help": "Random seed"})

def contain_question_mark(data):
    return data["target"][-1].rstrip() == "?"


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

    logger.info("saving dataset")

    data["train"].to_csv(os.path.join(args.data_dir, "train.csv"))
    data["validation"].to_csv(os.path.join(args.data_dir, "validation.csv"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
