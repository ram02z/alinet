import logging
import os
from dataclasses import dataclass, field
import re
import datasets
import pandas as pd
from datasets import (
    load_dataset,
    concatenate_datasets,
    DatasetDict,
)
import numpy as np
from transformers import set_seed, HfArgumentParser
from strenum import StrEnum
import unicodedata

logger = logging.getLogger(__name__)


class Dataset(StrEnum):
    BASELINE = "baseline"
    BASELINE_NOISE = "baseline_noise"
    BASELINE_BALANCED = "baseline_balanced"


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


def categorise_dataset(data):
    target = data["target"].lower()
    if any(word in target for word in ["what"]):
        data["category"] = "description"
    elif any(
        word in target
        for word in [
            "where",
            "when",
            "who",
            "how many",
            "how much",
            "which",
            "how long",
        ]
    ):
        data["category"] = "recall"
    elif any(
        word in target
        for word in ["how did", "how does", "how do", "compute", "calculate"]
    ):
        data["category"] = "method"
    elif any(word in target for word in ["why"]):
        data["category"] = "explanation"

    return data


def remove_na_category(data):
    return data["category"] != "NA"


def reduce_category_size(dataset, reduceTo, category):
    filtered_dataset = dataset.filter(lambda d: d["category"] == category).select(
        range(reduceTo)
    )
    rest_dataset = dataset.filter(lambda d: d["category"] != category)

    return concatenate_datasets([filtered_dataset, rest_dataset])


def print_distribution(dataset):
    categories = ["method", "description", "explanation", "recall", "NA"]

    distributions = []
    for category in categories:
        category_ds = dataset.filter(lambda data: data["category"] == category)
        distribution_str = f"{category} distribution = {len(category_ds) / len(dataset) * 100}%, count = {len(category_ds)}"
        distributions.append(distribution_str)

    for d in distributions:
        print(d)


def stratify_dataset(dataset, reduceTo):
    categories = ["method", "description", "explanation", "recall"]

    for category in categories:
        dataset = reduce_category_size(dataset, reduceTo, category)

    return dataset


def fix_encoding_errors(data):
    # This pattern matches one or more digits followed by an accented 'a'
    pattern = r"(\d+)Â"

    # See analysis in narrativeqa_encoding.ipynb
    data["source"] = (
        data["source"]
        .replace("â\x80\x94", ", ")
        .replace("Â\xa0â\x80\x93", " -")
        .replace("â\x80\x93", "-")
        .replace("â\x80\x99", "'")
        .replace("â\x80\x9d", "")
        .replace("â\x80\x9c", "")
        .replace("Ă˛", "")
        .replace("Ă\x89", "e")
        .replace("ÂŁ", "$")
        .replace("â\x80\x89", "")
        .replace("Ĺ\x8d", "o")
        .replace("â\x82Ź", "€")
    )
    data["source"] = re.sub(pattern, r"\1", data["source"])

    data["target"] = (
        data["target"]
        .replace("â\x80\x94", ", ")
        .replace("Â\xa0â\x80\x93", " -")
        .replace("â\x80\x93", "-")
        .replace("â\x80\x99", "'")
        .replace("â\x80\x9d", "")
        .replace("â\x80\x9c", "")
        .replace("Ă˛", "")
        .replace("Ă\x89", "e")
        .replace("ÂŁ", "$")
        .replace("â\x80\x89", "")
        .replace("Ĺ\x8d", "o")
        .replace("â\x82Ź", "€")
    )
    data["target"] = re.sub(pattern, r"\1", data["target"])

    return data


def add_dataset_name(data, name):
    data["dataset"] = name

    return data


def interleave_datasets(datasets):
    # To interleave the datasets, we concatenate them and then we re-order the indices
    concatenated_datasets = concatenate_datasets(datasets)

    # Build the indices to pass to .select()
    lengths = [len(dset) for dset in datasets]
    offsets = np.cumsum([0] + lengths[:-1])

    # Oversampling situation with cycling between each sources
    # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 5, 9, 0, 6, 10, 1, 3, 11]
    # Note that we have 5 examples per dataset with a rolling window since the longest dataset has 5 samples

    # Reasoning behind the following operation: for each dataset indices (i.e column) repeat the indices to have max_length indices per dataset
    # For example, if the max_length is 5 and the i-th dataset has 3 samples, the i-th column will be [0,1,2,0,1]
    indices = np.mod(
        np.arange(max(lengths)).reshape(-1, 1), np.array(lengths).reshape(1, -1)
    )
    # We have to keep the indices to their respective dataset offsets and to flatten to effectively interleave the datasets
    indices = (indices + offsets).flatten()
    # Keep unique indices based on order of appearance
    indices = pd.unique(indices).tolist()

    return concatenated_datasets.select(indices)


def main():
    parser = HfArgumentParser((GenerateDatasetArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    logger.info("loading datasets")
    if args.dataset == Dataset.BASELINE:
        data = (
            load_dataset("squad")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )
    elif args.dataset == Dataset.BASELINE_NOISE:
        squad_data = (
            load_dataset("squad")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )
        spoken_squad_data = (
            load_dataset("alinet/spoken_squad")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )
        train_data = concatenate_datasets(
            [squad_data["train"], spoken_squad_data["train"]]
        )
        valid_data = concatenate_datasets(
            [squad_data["validation"], spoken_squad_data["validation"]]
        )
        data = DatasetDict({"train": train_data, "validation": valid_data})
    elif args.dataset == Dataset.BASELINE_BALANCED:
        squad_data = (
            load_dataset("squad", trust_remote_code=True)
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .map(add_dataset_name, fn_kwargs={"name": "squad"})
        )

        adversarial_data = (
            load_dataset("adversarial_qa", "adversarialQA", trust_remote_code=True)
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .map(add_dataset_name, fn_kwargs={"name": "adversarial"})
        )

        narrative_data = (
            load_dataset("narrativeqa", trust_remote_code=True)
            .select_columns(["document", "question"])
            .map(
                lambda x: {
                    "document": x["document"]["summary"]["text"],
                    "question": x["question"]["text"],
                }
            )
            .rename_columns({"document": "source", "question": "target"})
            .map(fix_encoding_errors)
            .map(add_dataset_name, fn_kwargs={"name": "narrative"})
        )

        fairytale_data = (
            load_dataset("GEM/FairytaleQA", trust_remote_code=True)
            .filter(lambda x: x["ex_or_im"] == "explicit")
            .select_columns(["content", "target"])
            .rename_columns({"content": "source"})
            .map(add_dataset_name, fn_kwargs={"name": "fairytale"})
        )

        sciq_data = (
            load_dataset("sciq", trust_remote_code=True)
            .select_columns(["support", "question"])
            .rename_columns({"support": "source", "question": "target"})
            .filter(lambda x: x["source"] != "")
            .map(add_dataset_name, fn_kwargs={"name": "sciq"})
        )

        train_dataset = interleave_datasets(
            [
                squad_data["train"],
                adversarial_data["train"],
                narrative_data["train"],
                fairytale_data["train"],
                sciq_data["train"],
            ],
        )

        validate_dataset = interleave_datasets(
            [
                squad_data["validation"],
                concatenate_datasets(
                    [adversarial_data["validation"], adversarial_data["test"]]
                ),
                concatenate_datasets(
                    [narrative_data["validation"], narrative_data["test"]]
                ),
                concatenate_datasets(
                    [fairytale_data["validation"], fairytale_data["test"]]
                ),
                concatenate_datasets([sciq_data["validation"], sciq_data["test"]]),
            ],
        )

        train_dataset = train_dataset.add_column(
            "category", ["NA"] * len(train_dataset)
        )
        train_dataset = (
            train_dataset.filter(contain_question_mark)
            .map(normalise)
            .map(categorise_dataset)
            .filter(remove_na_category)
        )

        validate_dataset = validate_dataset.add_column(
            "category", ["NA"] * len(validate_dataset)
        )
        validate_dataset = (
            validate_dataset.filter(contain_question_mark)
            .map(normalise)
            .map(categorise_dataset)
            .filter(remove_na_category)
        )

        train_dataset = stratify_dataset(train_dataset, 4122)
        validate_dataset = stratify_dataset(validate_dataset, 1060)

        data = DatasetDict({"train": train_dataset, "validation": validate_dataset})

    logger.info("saving dataset")

    data["train"].to_csv(os.path.join(args.data_dir, "train.csv"))
    data["validation"].to_csv(os.path.join(args.data_dir, "validation.csv"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
