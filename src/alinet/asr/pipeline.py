import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import warnings
from pydub import AudioSegment
import numpy as np
from alinet.asr import Model


def pydub_to_np(audio_segment: AudioSegment) -> np.ndarray:
    """
    Converts an AudioSegment object to a normalized 1D NumPy array of type float32.
    The function ensures the audio is in mono (single channel), 16 kHz frame rate,
    and 16-bit sample width before conversion. The audio values are normalized
    to the range [-1.0, 1.0], suitable for 16-bit audio.
    """
    if audio_segment.frame_rate != 16000:
        audio_segment = audio_segment.set_frame_rate(16000)
    if audio_segment.sample_width != 2:
        audio_segment = audio_segment.set_sample_width(2)
    if audio_segment.channels != 1:
        audio_segment = audio_segment.set_channels(1)

    arr = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0

    return arr


class ASRPipeline:
    def __init__(self, model_id=Model.DISTIL_MEDIUM):
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._torch_dtype = (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self._device)

        self.processor = AutoProcessor.from_pretrained(model_id)

    def __call__(
        self,
        file_path: str,
        batch_size=16,
        chunk_length=15,
    ) -> (list[dict[str, str | tuple[float, float]]], float):
        """
        :param file_path: path of video/audio file
        :param batch_size: the size of each batch
        :param chunk_length: the input length for each chunk
        :return: transcript chunks with chunk-level timestamps and transcript duration
        """
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
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=chunk_length,
            batch_size=batch_size,
            torch_dtype=self._torch_dtype,
            device=self._device,
        )

        result = pipe(samples, return_timestamps=True)

        return result["chunks"], audio.duration_seconds


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="audio/video file")
    args = parser.parse_args()
    pipe = ASRPipeline()
    chunks, duration = pipe(file_path=args.file_path)
    data = {"chunks": chunks, "duration": duration}
    with open(datetime.now().strftime("%Y%m%d_%H%M%S.json"), "w") as f:
        json.dump(data, f)
