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

The dataset for the baseline system is generated using the `generate_dataset.py` script. 

The dataset combines the following sources:

- [SquAD 1.1](https://arxiv.org/abs/1606.05250)
- [AdversarialQA](https://doi.org/10.1162/tacl_a_00338)
- [NarrativeQA](https://arxiv.org/abs/1712.07040)
- [FairyTaleQA](https://arxiv.org/abs/2203.13947)

The script will output the combined CSV file in the `data` directory.

Example usage:

```shell
python data/generate_dataset.py \
    --remove_duplicate_context \
    --seed 42
```

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
    --train_csv_file path/to/train/data.csv \
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
export TASK_NAME="t5-base"

python train.py \
    --data_dir path/to/processed/data/dir \
    --output_dir path/to/output/dir \
    --model_id_or_path t5-base \
    --model_type t5 \
    --task_name $TASK_NAME \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --logging_steps 100
```

To report to [W&B](https://wandb.ai/), pass the `--report_to wandb` argument and set the following environment variables:

```shell
export WANDB_PROJECT="mmqg"
export WANDB_API_KEY="..."
export WANDB_WATCH="all"
export WANDB_LOG_MODEL="end"
```

For more information about the environment variables, refer to the [W&B documentation](https://docs.wandb.ai/guides/track/environment-variables).
