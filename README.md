# AliNET

> AliNET utilizes video lectures and presentation slides to generate high-quality questions pertinent to the provided educational content

## Installation

AliNET requires the following dependencies:

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
python main.py --video path/to/video/file.mp4
```

This will output all the generated questions.

To filter the generated questions using the original lecture slides, use the slides path:

```shell
# If no threshold is given, the default (0.5) is used
python main.py --video path/to/video/file.mp4 --slides path/to/slides.pdf
```
## Fine-tuning

See `/scripts` directory to learn how to generate and prepare the datasets, train the model and evaluate its performance.
