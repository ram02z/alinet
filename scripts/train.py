import logging
from dataclasses import dataclass, field
import datasets
from datasets import load_from_disk
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
)

from question_generation import ModelType

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    pretrained_model_name: str = field(
        default="google/t5-v1_1-base",
        metadata={"help": "Model identifier from huggingface.co/models"},
    )
    model_type: ModelType = field(
        default=ModelType.T5,
        metadata={"help": "Model type to use for training"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "Path of directory with cached dataset and tokenizer"},
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info("loading datasets")

    dataset = load_from_disk(data_args.data_dir)
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    logger.info("finished loading datasets")

    logger.info(f"training args: {training_args}")

    set_seed(training_args.seed)

    logger.info("loading pretrained model")

    tokenizer = AutoTokenizer.from_pretrained(data_args.data_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.pretrained_model_name)

    logger.info("finished loading pretrained model")

    # Initialise our DataCollator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id
    )

    # Initialise our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    logger.info("starting training")

    trainer.train()

    logger.info("finished training")

    trainer.save_model()
    logger.info(f"saved model to {training_args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
