# MMQG

MMQG: Is a multi-modal question generation system that utilizes video lectures and presentation slides to generate high-quality questions pertinent to the provided educational content. While initially designed for university professors, its applicability spans all academic levels. The primary aim is to assist teaching staff in efficiently creating instructional material, enabling students to self-assess their understanding of the content

# Installation
MMQG requires the following dependencies:

**Python 3.10** or later version:
- For installation visit: https://www.python.org/downloads/

**Poetry**, a python package manager for installation:
- For installation visit: https://python-poetry.org/docs/#installing-with-pipx

**FFmpeg**, software for handling of video and audio files
- Installation using **homebrew**:

```
brew install ffmpeg
```

# Usage
The system can be run by feeding it the path to the desired lecture video and the corresponding slides, with the file formats being **.mp4** and **PDF**, respectively:
```
python main.py path/to/video/file.mp4 path/to/slides.pdf
```
The output will be all the generated questions.

