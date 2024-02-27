# AliNET

> AliNET utilizes video lectures to generate high-quality questions pertinent to the provided educational content

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

To filter the generated questions using the lecture slides (from the video), you
can provide a similarity threshold and a retention rate (filtering) threshold:

```sh
# If no threshold is given, the default (0.5) is used for both
python main.py --video path/to/video/file.mp4 --similarity_threshold 0.6 --filtering_threshold 0.4
```

To save the video clips corresponding to the system-generated chunk, you can provide a directory path for the clips to be saved to:

```sh
# If no video clips path is provided, they won't be saved.
python main.py --video path/to/video/file.mp4 --video_clips_path directory/to/save/clips/to
```
## Fine-tuning

See `/scripts` directory to learn how to generate and prepare the datasets, train the model and evaluate its performance.
