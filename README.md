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
## Dataset preparation, Training and Evaluation

See `/scripts` directory to learn how to generate and prepare the datasets, train the model and evaluate its performance.