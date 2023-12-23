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

### Data preparation

The data processor script, `prepare_data.py`, expects the training data to be in the following format:

```csv
source,target
<context>,<question>
```

The data processor will process and cache the dataset in the 'torch' format and cache the tokenizer used to process the dataset.
The output of the files will be in the following format:

```shell
# Cached datasets
{split}_data_{model_type}.safetensors
# Cached tokenizer
tokenizer_{model_type}.json
```

Example usage:

```shell
python prepare_data.py \
    --train_csv_file path/to/train/data.csv \
    --valid_csv_file path/to/valid/data.csv \
    --model_type t5 \
    --max_source_length 512 \
    --max_target_length 32
```

### Training

The training script, `train.py`, expects the cached dataset and tokenizer, and model id and type to be passed as arguments.
To see the full list of training arguments, run `python train.py --help` or refer to HugginFace's [TrainerArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) documentation.

Data batches dynamically to fit the GPU memory by using `DataCollatorForSeq2Seq`.

Example usage:

```shell
export TASK_NAME="t5-base"

python train.py \
    --train_file_path path/to/train/data.safetensors \
    --valid_file_path path/to/valid/data.safetensors \
    --tokenizer path/to/tokenizer.json \
    --model_id_or_path t5-base \
    --model_type t5 \
    --task_name $TASK_NAME \
    --output_dir path/to/output/dir \
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
