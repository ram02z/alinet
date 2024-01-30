# MMQG

> Multi-modal question generation (MMQG) utilizes video lectures and presentation slides to generate high-quality questions pertinent to the provided educational content

## Installation

MMQG requires the following dependencies:

- Python 3.10
- Poetry
- FFmpeg

Using `brew` package manager as an example:

```shell
# Install system dependencies
brew install python@3.10 ffmpeg pipx

# Install poetry
pipx ensurepath
pipx install poetry

# Install project dependencies
poetry install
```

The virtual environment will be created locally in the `.venv` directory.

PyTorch with CUDA support will be only be installed on Linux environments.

## Usage

```sh
python main.py path/to/video/file.mp4
```

This will output all the generated questions.

To filter the generated questions using the original lecture slides, pass the slides path and the similarity threshold:

```shell
# If no threshold is given, the default (0.5) is used
python main.py path/to/video/file.mp4 path/to/slides.pdf --threshold 0.6
```

## Fine-tuning

### Dataset generation

The datasets used to train and validate the models can be generated using the `generate_dataset.py` script.

Datasets used:

- `baseline` (SQuAD 1.1)
- `baseline_noise` (SQuAD 1.1 + Spoken-SQuAD)
- `baseline_balanced` (SQuAD 1.1 + Adversarial_QA + NarrativeQA + FairytaleQA + SciQ + comparisonQA)

Example usage:

```shell
python data/generate_dataset.py \
    --dataset {baseline,baseline_noise} \
    --data_dir data \
    --seed 42
```

The script will output `train` and `validation` splits as CSV files to the
`data` directory when dataset is set to `baseline`.

### Data preparation

The data processor script, `prepare_data.py`, expects the training data to be in the following format:

```csv
source,target
<context>,<question>
```

The data processor will process and cache the dataset, and save the tokenizer in the specified output directory (default is `./data/`).

Example usage:

```shell
python prepare_data.py \
    --data_dir path/to/data/dir \
    --output_dir path/to/output/dir \
    --model_type t5 \
    --max_source_length 512 \
    --max_target_length 32 \
    --seed 42
```

### Training

The training script, `train.py`, expects the processed data directory to contain dataset splits and the `tokenizer_config.json` file (see [Data preparation usage](#data-preparation)).

To see the full list of training arguments, run `python train.py --help` or refer to HuggingFace's [TrainerArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) documentation.

Example usage:

```shell
export RUN_NAME="t5-base"

python train.py \
    --data_dir path/to/processed/data/dir \
    --output_dir path/to/output/dir \
    --pretrained_model_name t5-base \
    --model_type t5 \
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
python eval.py \
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
