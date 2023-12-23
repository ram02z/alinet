import logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser, T5Tokenizer, BartTokenizer, PreTrainedTokenizer
from safetensors.torch import save_file

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
    valid_csv_file: str = field(
        metadata={
            "help": "Path of CSV file with validation data with source and target column"
        }
    )
    model_type: ModelType = field(
        default=ModelType.T5, metadata={"help": "Model type to use for training"}
    )
    max_source_length: int = field(
        default=512, metadata={"help": "Maximum input length for the source text"}
    )
    max_target_length: int = field(
        default=32, metadata={"help": "Maximum input length for the target text"}
    )


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
                self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
            case ModelType.BART:
                self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, dataset: datasets.Dataset) -> (datasets.Dataset, PreTrainedTokenizer):
        """
        :param dataset: dataset to process
        """
        tokenized_dataset = dataset.map(self._convert_to_features, batched=True)
        columns = ["input_ids", "labels", "attention_mask"]
        tokenized_dataset.set_format(type="torch", columns=columns)

        return tokenized_dataset, self.tokenizer

    def _convert_to_features(self, x):
        input_encodings = self.tokenizer.batch_encode_plus(
            x["source"],
            max_length=self.max_target_length,
            truncation=True,
            add_special_tokens=True,
        )
        target_encodings = self.tokenizer.batch_encode_plus(
            x["target"],
            max_length=self.max_source_length,
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

    train_dataset = datasets.load_dataset(args.train_csv_file)
    valid_dataset = datasets.load_dataset(args.valid_csv_file)

    process, tokenizer = DataProcessor(
        args.model_type, args.max_source_length, args.max_target_length
    )

    train_dataset = process(train_dataset)
    valid_dataset = process(valid_dataset)

    train_file_path = f"train_data_{args.model_type}.safetensors"
    save_file(train_dataset, train_file_path)
    logger.info(f"saved train dataset to {train_file_path}")

    valid_file_path = f"valid_data_{args.model_type}.safetensors"
    save_file(valid_dataset, valid_file_path)
    logger.info(f"saved validation dataset to {valid_file_path}")

    tokenizer_file_path = f"tokenizer_{args.model_type}"
    tokenizer.save_pretrained(tokenizer_file_path)
    logger.info(f"saved tokenizer to {tokenizer_file_path}")


if __name__ == "__main__":
    main()
