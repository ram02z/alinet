## Fine-tuning

### Dataset generation

The datasets used to train and validate the models can be generated using the `generate_dataset.py` script. 

Datasets used:
- `baseline` (SQuAD 1.1)
- `baseline_noise` (SQuAD 1.1 + Spoken-SQuAD)
- `baseline_balanced` (SQuAD 1.1 + AdversarialQA + NarrativeQA + SciQ)

Example usage:

```shell
python scripts/generate_dataset.py \
    --dataset {baseline,baseline_noise,baseline_balanced} \
    --output_dir path/to/data/dir \
    --model_type t5 \
    --max_source_length 512 \
    --max_target_length 32 \
    --seed 42
```

The script will output `train` and `validation` splits as CSV files to the
`path/to/data/dir` directory. The tokenized dataset will also be saved in the
same directory along with the original tokenizer.

#### Coreference resolution

The datasets can modified to reduce ambiguity in the questions using the `coreference_resolution.py` script.

```shell
python scripts/coreference_resolution.py path/to/data/file.csv
```

#### Spoken data augmentation

The dataset can also be modified to add spoken noise to the contexts using the `augment_dataset.py` script.

The script uses the [mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) model for text-to-speech (TTS) and the [distil-whisper](https://huggingface.co/distil-whisper/distil-large-v2) for automatic speech recognition (ASR).

```shell
python scripts/augment_dataset.py path/to/data/file.csv
```

### Training

#### Pre-training

> BART

The pre-training script, `run_bart_dlm_flax.py`, has been copied from
[huggingface/transformers](https://github.com/huggingface/transformers/blob/831bc25d8fdb85768402f772cf65cc3d7872b211/examples/flax/language-modeling/run_bart_dlm_flax.py)
and adapted for this project.

The following changes have been made:
- Introduced `text_column_name` argument to specify text column
- Integrated [W&B](https://wandb.ai/) Python client library
- Removed HuggingFace telemetry

Make sure you train the tokenizer on the text data and create the model
configuration before running the script. See [here](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#bart-denoising-language-modeling) for more detail.

Example usage:

```shell
python run_bart_dlm_flax.py \
    --output_dir="./pubmed-bart-base" \
    --config_name="./pubmed-bart-base" \
    --tokenizer_name="./pubmed-bart-base" \
    --dataset_name="alinet/pubmed_qa" \
    --text_column_name="context" \
    --validation_split_percentage="1" \
    --max_seq_length="1024" \
    --per_device_train_batch_size="4" \
    --per_device_eval_batch_size="4" \
    --learning_rate="1e-4" \
    --warmup_steps="2000" \
    --overwrite_output_dir \
    --logging_steps="500" \
    --save_steps="2000" \
    --eval_steps="2000"
```

To report to [W&B](https://wandb.ai/), set the following environment variables:

```shell
export WANDB_API_KEY="..."
export WANDB_NAME="pubmed-bart-base"
```

#### Fine-tuning

The fine-tuning script, `train.py`, expects the processed data directory to contain dataset splits and the `tokenizer_config.json` file (see [Dataset generation](#dataset-generation)).

To see the full list of training arguments, run `python train.py --help` or refer to HuggingFace's [TrainerArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) documentation.

Example usage:

```shell
export RUN_NAME="bart-base"

python scripts/train.py \
    --data_dir path/to/processed/data/dir \
    --output_dir path/to/output/dir \
    --pretrained_model_name facebook/bart-base \
    --model_type bart \
    --run_name $RUN_NAME \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --seed 42 \
    --do_train \
    --do_eval \
    --logging_steps 100
```

To report to [W&B](https://wandb.ai/), pass the `--report_to wandb` argument to
`train.py` and set the following environment variables:

```shell
export WANDB_PROJECT="mmqg"
export WANDB_API_KEY="..."
export WANDB_WATCH="all"
export WANDB_LOG_MODEL="end"
```

For more information about the environment variables, refer to the [W&B documentation](https://docs.wandb.ai/guides/track/environment-variables).

### Evaluation

The fine-tuned model's performance can be evaluated using the `eval.py` script.

The script supports the following evaluation datasets:
- `reading_comprehension` (MRQA 2019 test split)
- `spoken_noise` (Spoken-SQuAD WER54 test split)

The script also supports the following evaluation metrics:
- [BERTScore](https://arxiv.org/abs/1904.09675)


Example usage:

```shell
python scripts/eval.py \
       --pretrained_model_name_or_path path/to/model \
       --dataset {reading_comprehension,spoken_noise}
       --evaluation_module bertscore \
       --max_length 32 \
       --num_beams 4
```

To push the metric results to the HuggingFace Hub, you need authenticate first:

```shell
huggingface-cli login
```

Ensure that `--push_to_hub` is passed to the `eval.py` script.

The script will output the evaluation results to the `results` directory.
