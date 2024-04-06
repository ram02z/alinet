# AliNET

> AliNET utilizes video lectures to generate high-quality questions pertinent to the provided educational content

## Installation

AliNET requires the following dependencies:

- Python 3.10
- Poetry
- Tesseract
- FFmpeg

Using `brew` package manager as an example:

```shell
# Install system dependencies
brew install python@3.10 ffmpeg tesseract pipx
# Install poetry
pipx ensurepath
pipx install poetry

# Install project dependencies
poetry install
```

The virtual environment will be created locally in the `.venv` directory.

PyTorch with CUDA support will be only be installed on Linux environments.

## Usage

### Web App

To start the server, run the following command:

```sh
python main.py
```

To start the React application, run the following commands:

```sh
cd web_app
npm run dev
```

### Command Line Interface

This will generate all questions and the filtered video clips. Used for our human evaluation.

```sh
python cli.py \
  --video path/to/video/file.mp4 \
  --similarity_threshold 0 \
  --output_dir_path path/to/output/directory
```

This will save the questions and all video clips to `path/to/output/directory`.

To filter the generated questions using the lecture slides (from the video), you
can provide a similarity threshold and a retention rate (filtering) threshold:

```sh
# If no threshold is given, the default (0.5) is used for both
python cli.py \
  --video path/to/video/file.mp4 \
  --similarity_threshold 0.6 \
  --filtering_threshold 0.4 \
  --output_dir_path path/to/output/directory
```

This will save only the video clips associated with the questions that have a similarity score greater than `0.6` and a retention rate greater than `0.4`.

## Fine-tuning

See `/scripts` directory to learn how to generate and prepare the datasets, train the model and evaluate its performance.
