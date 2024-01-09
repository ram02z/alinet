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
The script will output three CSV files in the `data` directory for training, validation, and testing.

The dataset combines the following sources:

- [SquAD 1.1](https://arxiv.org/abs/1606.05250)
- [AdversarialQA](https://doi.org/10.1162/tacl_a_00338)
- [NarrativeQA](https://arxiv.org/abs/1712.07040)
- [FairyTaleQA](https://arxiv.org/abs/2203.13947)

Example usage:

```shell
python data/generate_dataset.py \
       --remove_duplicate_context \
       --seed 42
```

### Evaluation

The fine-tuned model's performance can be evaluated using the `eval.py` script. 

The script supports the following evaluation metrics:
- [BERTScore](https://arxiv.org/abs/1904.09675)

The script will output the evaluation results to the `results` directory.

Example usage:

```shell
python eval.py \
       --pretrained_model_name_or_path path/to/model \
       --evaluation_module bertscore \
       --max_length 32 \
       --num_beams 4
```

To push the metric results to the HuggingFace Hub, you need authenticate first:

```shell
huggingface-cli login
```

Ensure that `--push_to_hub` is passed to the `eval.py` script.
