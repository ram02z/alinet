import logging
from dataclasses import dataclass, field

from datasets import DatasetDict
from transformers import HfArgumentParser, set_seed, T5Tokenizer, BartTokenizer

from strenum import StrEnum
import datasets

logger = logging.getLogger(__name__)


class ModelType(StrEnum):
    BART = "bart"
    T5 = "t5"


@dataclass
class PrepareDataArguments:
    train_csv_file: str = field(
        metadata={
            "help": "Path of CSV file with training data with source and target column"
        }
    )
    output_dir: str = field(default="data", metadata={"help": "Output directory"})
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
        match model_type:
            case ModelType.T5:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    "t5-base", model_max_length=max_source_length
                )
            case ModelType.BART:
                self.tokenizer = BartTokenizer.from_pretrained(
                    "facebook/bart-base", model_max_length=max_source_length
                )
            case _:
                raise ValueError(f"Unsupported model type {model_type}")
        self.max_source_length = max_source_length

    def __call__(self, dataset: datasets.Dataset, seed: int):
        """
        :param dataset: dataset to process
        """
        stratify_column = "dataset"
        dataset = dataset.class_encode_column(stratify_column)
        train_testvalid = dataset.train_test_split(
            test_size=0.1, stratify_by_column=stratify_column, seed=seed
        )
        test_valid = train_testvalid["test"].train_test_split(
            test_size=0.5, stratify_by_column=stratify_column, seed=seed
        )

        train_test_valid_dataset = DatasetDict(
            {
                "train": train_testvalid["train"],
                "test": test_valid["test"],
                "valid": test_valid["train"],
            }
        )
        tokenized_dataset = train_test_valid_dataset.map(
            self._convert_to_features,
            batched=True,
            remove_columns=["source", "target", stratify_column],
        )
        columns = ["input_ids", "labels", "attention_mask"]
        tokenized_dataset.set_format(type="torch", columns=columns)

        return tokenized_dataset

    def _convert_to_features(self, x):
        input_encodings = self.tokenizer.batch_encode_plus(
            x["source"],
            truncation=True,
            add_special_tokens=True,
        )
        target_encodings = self.tokenizer.batch_encode_plus(
            x["target"],
            truncation=True,
            add_special_tokens=True,
        )

        encodings = {
            "input_ids": input_encodings["input_ids"],
            "labels": target_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
        }

        return encodings


def main():
    parser = HfArgumentParser((PrepareDataArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    logger.info("loading dataset")

    train_dataset = datasets.load_dataset(
        "csv", data_files=args.train_csv_file, split="train"
    )

    logger.info("finished loading dataset")

    process = DataProcessor(
        args.model_type, args.max_source_length, args.max_target_length
    )

    logger.info("processing dataset")

    dataset_dict = process(train_dataset, args.seed)

    logger.info("finished processing dataset")

    dataset_dict.save_to_disk(args.output_dir)
    logger.info("saved datasets")

    tokenizer = process.tokenizer
    tokenizer.save_pretrained(args.output_dir)
    logger.info("saved tokenizer")


if __name__ == "__main__":
    main()
