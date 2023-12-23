import logging
from dataclasses import dataclass, field

from safetensors.torch import load_file
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)

from prepare_data import ModelType

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    cached_tokenizer_path: str = field(
        metadata={"help": "Path for cached tokenizer (must be same as model type)"}
    )
    pretrained_model_name: str = field(
        default="t5-base",
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

    train_file_path: str = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: str = field(
        metadata={"help": "Path for cached validation dataset"}
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"training args: {training_args}")

    set_seed(training_args.seed)

    logger.info("loading tokenizer and model")

    tokenizer = PreTrainedTokenizer.from_pretrained(model_args.cached_tokenizer_path)
    config = AutoConfig.from_pretrained(
        model_args.pretrained_model_name, decoder_start_token_id=tokenizer.pad_token_id
    )
    model = AutoModelForSeq2SeqLM(config).from_pretrained(
        model_args.pretrained_model_name
    )

    logger.info("finished loading tokenizer and model")

    logger.info("loading datasets")

    train_dataset = load_file(data_args.train_file_path)
    valid_dataset = load_file(data_args.valid_file_path)

    logger.info("finished loading datasets")

    # Initialise our DataCollator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id
    )

    # Initialise our TrainingArguments
    extra_training_args = dict(
        run_name=model_args.pretrained_model_name,
        report_to="wandb",
        prediction_loss_only=True,
    )
    args_dict = {**training_args.to_dict(), **extra_training_args}
    args = TrainingArguments(**args_dict)

    # Initialise our Trainer
    trainer = Seq2SeqTrainer(
        models=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    logger.info("starting training")
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
