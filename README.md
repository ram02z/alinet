# MMQG

> Multi-modal question generation (MMQG) utilizes video lectures and presentation slides to generate high-quality questions pertinent to the provided educational content

## Installation
MMQG requires the following dependencies:

- Python >= 3.10
- Poetry
- FFmpeg

Using `brew` package manager as an example:

```sh
# Install system dependencies
brew install python@3.10 ffmpeg pipx

# Install poetry
pipx ensurepath
pipx install poetry

# Install project dependencies
poetry install
```

The virtual environment will be created locally in the `.venv` directory.

## Usage
The system can be run by feeding it the path to the desired lecture video and the corresponding slides, with the file formats being **.mp4** and **PDF**, respectively:
```sh
python main.py path/to/video/file.mp4 path/to/slides.pdf
```
The output will be all the generated questions.

