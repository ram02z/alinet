import logging
import argparse
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    VitsModel,
    AutoTokenizer,
    pipeline,
)
import torch
import numpy as np
import os

import datasets
from datasets import load_dataset
from evaluate import load

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "distil-whisper/distil-large-v2", torch_dtype=torch_dtype, use_safetensors=True
).to(device)
asr_processor = AutoProcessor.from_pretrained("distil-whisper/distil-large-v2")
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=asr_processor.tokenizer,
    feature_extractor=asr_processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)
wer = load("wer")

cache = {}

def augment_dataset(example):
    source = example["source"]

    if source in cache:
        text = cache[source]
    else:
        inputs = tts_tokenizer(source, return_tensors="pt").to(device)
        with torch.no_grad():
            output = tts_model(**inputs)
        np_array = output.to(device).numpy()
        if np_array.ndim > 1:
            np_array = np.mean(np_array, axis=0)

        result = asr_pipe(np_array, batch_size=1)
        text = result["text"]
        cache[source] = text

    example["source"] = text
    return example


def get_new_file_path(fp):
    dir = os.path.dirname(fp)

    filename_with_ext = os.path.basename(fp)
    filename, file_ext = os.path.splitext(filename_with_ext)

    new_fn = f"{filename}-augmented{file_ext}"

    new_fp = os.path.join(dir, new_fn)

    return new_fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", help="Path to CSV file to run data augmentation on"
    )

    args = parser.parse_args()
    data = load_dataset("csv", data_files=args.file_path, split="train")

    references = list(data["source"])
    data = data.map(augment_dataset)
    predictions = list(data["source"])
    wer_score = wer.compute(predictions=predictions, references=references)
    logger.info(f"WER: {wer_score}")


    data.to_csv(get_new_file_path(args.file_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
