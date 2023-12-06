import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import warnings
from pydub import AudioSegment
import numpy as np


def pydub_to_np(audio: AudioSegment) -> np.ndarray:
    """
    Converts an AudioSegment object to a normalized 1D NumPy array of type float32.
    The audio values are normalized to the range [-1.0, 1.0], suitable for 16-bit audio.
    """
    return (
        np.frombuffer(audio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0
    )


class ASR:
    def __init__(self):
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._torch_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )

    def transcribe(
        self, file_path: str, batch_size=16, chunk_length=15
    ) -> list[dict[str | tuple[float, float]]]:
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension[1:]

        match file_extension:
            case "mp3":
                audio = AudioSegment.from_mp3(file_path)
            case "mp4":
                audio = AudioSegment.from_file(file_path, "mp4")
            case _:
                warnings.warn(
                    f"Unsupported file extension '{file_extension}' processed."
                )
                audio = AudioSegment.from_file(file_path, file_extension)

        samples = pydub_to_np(audio)
        model_id = "distil-whisper/distil-medium.en"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self._device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=chunk_length,
            batch_size=batch_size,
            torch_dtype=self._torch_dtype,
            device=self._device,
        )

        result = pipe(samples, return_timestamps=True)

        return result["chunks"]


# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "file_path",
#     help="type the file path to your video here, e.g. sample_data/video.mp4",
# )
# args = parser.parse_args()
#
# # Create a temporary folder to store created audio files
# temp_dir = tempfile.TemporaryDirectory()
# print(temp_dir.name)
#
#
# # Default FFMPEG method of extracting audio from video and generating an mp3
# def convert_video_to_mp3(video_path, audio_path):
#     ffmpeg_cmd = [
#         "ffmpeg",
#         "-i",
#         video_path,
#         "-vn",
#         "-y",
#         audio_path,
#     ]  # overrides output files
#
#     try:
#         subprocess.run(ffmpeg_cmd, check=True)
#         print("converted")
#     except subprocess.CalledProcessError:
#         print("failed")
#
#
# audio__storage_path = temp_dir.name + "/audioOutput.mp3"
# convert_video_to_mp3(args.file_path, audio__storage_path)
#
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#
# model_id = "distil-whisper/distil-medium.en"
#
# # NOTE: Using Accelerate library enables us to set the parameters, low_cpu_mem_usage=True, use_safetensors=True
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id,
#     torch_dtype=torch_dtype,
# )
# model.to(device)
#
# processor = AutoProcessor.from_pretrained(model_id)
#
# transcriber = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,  # NOTE: Change the batch size to what works on your machine for development
#     torch_dtype=torch_dtype,
#     device=device,
# )
#
# result = transcriber(inputs=audio__storage_path, return_timestamps=True)
#
# print(result["text"])
#
# # Delete the temporary folder
# temp_dir.cleanup()

if __name__ == "__main__":
    asr = ASR()
    chunks = asr.transcribe("lecture_sample.mp4", chunk_length=15)
    print(chunks)
