import subprocess
import tempfile
import argparse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

parser = argparse.ArgumentParser()
parser.add_argument("file_path", help="type the file path to your video here, e.g. sample_data/video.mp4")
args = parser.parse_args()

# Create a temporary folder to store created audio files
temp_dir = tempfile.TemporaryDirectory()
print(temp_dir.name)

# Default FFMPEG method of extracting audio from video and generating an mp3
def convert_video_to_mp3(video_path, audio_path):
    ffmpeg_cmd = ["ffmpeg", "-i", video_path, "-vn", "-y", audio_path]  # overrides output files

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print("converted")
    except subprocess.CalledProcessError as e:
        print("failed")

audio__storage_path = temp_dir.name + "/audioOutput.mp3"
convert_video_to_mp3(args.file_path, audio__storage_path)



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