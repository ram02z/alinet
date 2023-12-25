import logging
import os
from dataclasses import dataclass, field

from datasets import load_dataset, concatenate_datasets
from transformers import set_seed, HfArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class GenerateDatasetArguments:
    seed: int = field(default=42, metadata={"help": "Random seed"})
    remove_duplicate_context: bool = field(
        default=True,
        metadata={"help": "Keep only one context for each question"},
    )


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

    squad_data = (
        load_dataset("squad", split="train+validation")
        .select_columns(["context", "question"])
        .rename_columns({"context": "source", "question": "target"})
    )
    adversarial_data = (
        load_dataset("adversarial_qa", "adversarialQA", split="train+validation+test")
        .select_columns(["context", "question"])
        .rename_columns({"context": "source", "question": "target"})
    )
    narrative_data = (
        load_dataset("narrativeqa", split="train+validation+test")
        .select_columns(["document", "question"])
        .map(
            lambda x: {
                "document": x["document"]["summary"]["text"],
                "question": x["question"]["text"],
            }
        )
        .rename_columns({"document": "source", "question": "target"})
    )
    fairytale_data = (
        load_dataset("GEM/FairytaleQA", split="train+validation+test")
        .filter(lambda x: x["ex_or_im"] == "explicit")
        .select_columns(["content", "target"])
        .rename_columns({"content": "source"})
    )

    logger.info("concatenating datasets")

    dataset = concatenate_datasets(
        [squad_data, adversarial_data, narrative_data, fairytale_data]
    )

    logger.info("filtering datasets")

    dataset = dataset.filter(contain_question_mark)
    if args.remove_duplicate_context:
        unique_sources = set()
        dataset = dataset.filter(
            contain_unique_question_context,
            fn_kwargs={"unique_sources": unique_sources},
        )
    dataset = dataset.map(normalise)

    logger.info("saving dataset")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    dataset.to_csv(os.path.join(script_dir, "dataset.csv"))


if __name__ == "__main__":
    main()
