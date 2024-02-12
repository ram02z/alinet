import logging
import os
from dataclasses import dataclass, field
import re
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import BartTokenizer, T5Tokenizer, set_seed, HfArgumentParser
from strenum import StrEnum
import unicodedata
from models import ModelType

logger = logging.getLogger(__name__)


class Dataset(StrEnum):
    BASELINE = "baseline"
    BASELINE_NOISE = "baseline_noise"
    BASELINE_BALANCED = "baseline_balanced"


@dataclass
class GenerateDatasetArguments:
    dataset: Dataset = field(metadata={"help": "Name of the dataset to use"})
    output_dir: str = field(default="data", metadata={"help": "Output data directory"})
    model_type: ModelType = field(
        default=ModelType.T5, metadata={"help": "Model type to use for training"}
    )
    max_source_length: int = field(
        default=512, metadata={"help": "Maximum input length for the source text"}
    )
    max_target_length: int = field(
        default=32, metadata={"help": "Maximum input length for the target text"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})


class DataProcessor:
    def __init__(
        self, model_type=ModelType.T5, max_source_length=512, max_target_length=32
    ):
        """
        :param model_type: ModelType enum
        :param max_source_length: maximum length of source text
        :param max_target_length: maximum length of target text
        """
        if model_type == ModelType.T5:
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        elif model_type == ModelType.BART:
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        else:
            raise ValueError(f"Unsupported model type {model_type}")
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, dataset: datasets.Dataset | datasets.DatasetDict):
        """
        :param dataset: dataset to process
        """
        tokenized_dataset = dataset.map(
            self._convert_to_features,
            batched=True,
            batch_size=2000,
        ).filter(self._filter_max_length)
        columns = ["input_ids", "labels", "attention_mask"]
        tokenized_dataset.set_format(type="torch", columns=columns)

        return tokenized_dataset

    def _filter_max_length(self, x):
        return (
            len(x["input_ids"]) <= self.max_source_length
            and len(x["labels"]) <= self.max_target_length
        )

    def _convert_to_features(self, x):
        input_encodings = self.tokenizer.batch_encode_plus(
            x["source"],
            add_special_tokens=True,
        )
        target_encodings = self.tokenizer.batch_encode_plus(
            x["target"],
            add_special_tokens=True,
        )

        encodings = {
            "input_ids": input_encodings["input_ids"],
            "labels": target_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
        }

        return encodings


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


def categorise_dataset(data):
    target = data["target"].lower()
    if any(word in target for word in ["what"]):
        data["category"] = "description"
    elif any(
        word in target
        for word in [
            "how did",
            "how does",
            "how do",
            "compute",
            "calculate",
            "how can",
            "how should",
            "how would",
            "how will",
            "how to",
        ]
    ):
        data["category"] = "method"
    elif any(
        word in target
        for word in [
            "where",
            "when",
            "who",
            "how",
            "which",
        ]
    ):
        data["category"] = "recall"
    elif any(word in target for word in ["why"]):
        data["category"] = "explanation"
    else:
        data["category"] = "NA"

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


def stratify_dataset(dataset):
    categories = ["method", "description", "explanation", "recall"]

    reduceTo = get_lowest_category_count(dataset, categories)

    for category in categories:
        dataset = reduce_category_size(dataset, reduceTo, category)

    return dataset


def get_lowest_category_count(dataset, categories):
    distributions = []

    for category in categories:
        category_ds = dataset.filter(lambda data: data["category"] == category)
        distribution = len(category_ds)
        distributions.append(distribution)

    return min(distributions)


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


def main():
    parser = HfArgumentParser((GenerateDatasetArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    process = DataProcessor(
        args.model_type, args.max_source_length, args.max_target_length
    )

    logger.info("loading datasets")
    if args.dataset == Dataset.BASELINE:
        data = (
            load_dataset("squad")
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
            .filter(contain_question_mark)
            .map(normalise)
        )
        data = process(data)
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
        data = DatasetDict(
            {"train": train_data, "validation": squad_data["validation"]}
        )
        data = process(data)
    elif args.dataset == Dataset.BASELINE_BALANCED:
        squad_data = (
            load_dataset("squad", trust_remote_code=True)
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
        )

        adversarial_data = (
            load_dataset("adversarial_qa", "adversarialQA", trust_remote_code=True)
            .select_columns(["context", "question"])
            .rename_columns({"context": "source", "question": "target"})
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
        )

        fairytale_data = (
            load_dataset("GEM/FairytaleQA", trust_remote_code=True)
            .filter(lambda x: x["ex_or_im"] == "explicit")
            .select_columns(["content", "target"])
            .rename_columns({"content": "source"})
        )

        sciq_data = (
            load_dataset("sciq", trust_remote_code=True)
            .select_columns(["support", "question"])
            .rename_columns({"support": "source", "question": "target"})
            .filter(lambda x: x["source"] != "")
        )

        train_dataset = concatenate_datasets(
            [
                squad_data["train"],
                adversarial_data["train"],
                narrative_data["train"],
                fairytale_data["train"],
                sciq_data["train"],
            ]
        )

        validate_dataset = concatenate_datasets(
            [
                squad_data["validation"],
                adversarial_data["validation"],
                adversarial_data["test"],
                narrative_data["validation"],
                narrative_data["test"],
                fairytale_data["validation"],
                fairytale_data["test"],
                sciq_data["validation"],
                sciq_data["test"],
            ]
        )

        train_dataset = (
            train_dataset.filter(contain_question_mark)
            .map(normalise)
            .map(categorise_dataset)
            .filter(remove_na_category)
        )

        validate_dataset = (
            validate_dataset.filter(contain_question_mark)
            .map(normalise)
            .map(categorise_dataset)
            .filter(remove_na_category)
        )

        train_dataset = process(train_dataset)
        validate_dataset = process(validate_dataset)

        train_dataset = stratify_dataset(train_dataset)
        validate_dataset = stratify_dataset(validate_dataset)

        data = DatasetDict({"train": train_dataset, "validation": validate_dataset})

    logger.info("saving text dataset")

    text_data = data.remove_columns(["input_ids", "labels", "attention_mask"])
    text_data["train"].to_csv(os.path.join(args.output_dir, "train.csv"))
    text_data["validation"].to_csv(os.path.join(args.output_dir, "validation.csv"))

    logger.info("saving tokenised dataset")
    tokenised_data = data.select_columns(["input_ids", "labels", "attention_mask"])
    tokenised_data.save_to_disk(args.output_dir)

    logger.info("saving tokenizer")
    tokenizer = process.tokenizer
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
