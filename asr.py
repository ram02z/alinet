import subprocess
import os


def convert_video_to_mp3(input, output):
    ffmpeg_cmd = ["ffmpeg", "-i", input, "-vn", "-y", output]  # overrides output files

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("converted")
    except subprocess.CalledProcessError as e:
        print("failed")

# Change the name of your videos into videoInput.mp4
# TODO: Implement a method of uploading a video instead of hardcoding path
convert_video_to_mp3("sample_data/videoInput.mp4", "sample_data/audioOutput.mp3")

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

# result = transcriber(inputs="audio.mp3", return_timestamps=True)["chunks"]
result = transcriber(inputs="audioOutput.mp3", return_timestamps=True)

print(result["text"])
