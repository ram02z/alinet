import sys
import os
import ffmpeg
import numpy as np
import subprocess
import tempfile

# Create a temporary folder to store created audio files
temp_dir = tempfile.TemporaryDirectory()
print(temp_dir.name)

if len(sys.argv) < 2:
    print("Please run the program again, and enter a file path to your video e.g. sample_data/video.mp4")
    exit()

video_path = sys.argv[1]
print("Video Path Specified: ", video_path)

if video_path[-3:] != "mp4":
    print("Please run the program again, and input a correct file name containing the mp4 file extension (.mp4)")
    exit()

# Default FFMPEG method of extracting audio from video and generating an mp3
def convert_video_to_mp3(video_path, audio_path):
    ffmpeg_cmd = ["ffmpeg", "-i", video_path, "-vn", "-y", audio_path]  # overrides output files

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("converted")
    except subprocess.CalledProcessError as e:
        print("failed")

audio__storage_path = temp_dir.name + "/audioOutput.mp3"
convert_video_to_mp3(video_path, audio__storage_path)

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

print("distil-whisper/medium-en")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-medium.en"

# NOTE: Using Accelerate library enables us to set the parameters, low_cpu_mem_usage=True, use_safetensors=True
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, 
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16, # NOTE: Change the batch size to what works on your machine for development 
    torch_dtype=torch_dtype,
    device=device,
)

result = transcriber(inputs=audio__storage_path, return_timestamps=True)

print(result["text"])

# Delete the temporary folder
temp_dir.cleanup()