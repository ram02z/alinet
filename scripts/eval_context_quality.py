from dataclasses import dataclass, field
import datasets
from transformers import HfArgumentParser, set_seed
import evaluate
import openpyxl
from openpyxl import Workbook

import json
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)
from alinet import chunking, qg, rag  # noqa: E402


def create_or_load_workbook(filename):
    try:
        workbook = openpyxl.load_workbook(filename)
    except FileNotFoundError:
        workbook = Workbook()
    return workbook


@dataclass
class EvaluateRAGArguments:
    lecture_name: str = field(
        metadata={"help": "Name of lecture to store data in excel worksheet"}
    )
    transcript_path: str = field(metadata={"help": "Path to transcript"})
    doc_paths: list[str] = field(
        metadata={"help": "List of document paths"},
    )
    top_k: int = field(
        default=1,
        metadata={"help": "Number of relevant contexts to retrieve"},
    )
    distance_threshold: float = field(
        default=0,
        metadata={"help": "Distance threshold to consider a context as relevant"},
    )
    collection_name: str = field(
        default="default",
        metadata={"help": "The name of the collection in the vectordb"},
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})


# Run the script on one lecture at a time

# One video
# Multiple Supplementary Material

# Similarity Threshold: 0
# Top_K: 1
# Max Token Limit: 32


def main():
    parser = HfArgumentParser((EvaluateRAGArguments))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)
    db = rag.Database()

    collection = db.create_collection()

    # Store documents in collection
    pdfs_bytes: list[bytes] = []
    for doc_path in args.doc_paths:
        with open(doc_path, "rb") as f:
            pdf_bytes = f.read()
        pdfs_bytes.append(pdf_bytes)

    db.store_documents(collection, pdfs_bytes=pdfs_bytes)

    # Path to transcript
    with open(args.transcript_path, "rb") as file:
        data = json.load(file)
        whisper_chunks = data["chunks"]
        duration = data["duration"]

    chunk_pipe = chunking.Pipeline(qg.Model.BALANCED_RESOLVED)
    transcript_chunks = chunk_pipe(whisper_chunks, duration, stride_length=32)

    source_texts = [chunk.text for chunk in transcript_chunks]

    sources_with_context = db.add_relevant_context_to_sources(
        source_texts=source_texts,
        collection=collection,
        top_k=args.top_k,
        distance_threshold=args.distance_threshold,
    )

    text_pairs = []

    for i, text_with_context in enumerate(sources_with_context):
        original_text = source_texts[i]
        context_only_text = text_with_context.split(original_text, 1)[-1].strip()
        text_pairs.append((original_text, context_only_text))

    # Create or load the workbook
    workbook = create_or_load_workbook("source_vs_rag.xlsx")

    # Create a new worksheet with the lecture name
    worksheet_name = args.lecture_name
    worksheet = workbook.create_sheet(worksheet_name)

    # Write the header row
    worksheet["A1"] = "Source Text"
    worksheet["B1"] = "Context"

    # Write the data rows
    for row_num, pair in enumerate(text_pairs, start=2):
        worksheet.cell(row=row_num, column=1, value=pair[0])
        worksheet.cell(row=row_num, column=2, value=pair[1])

    # Save the workbook
    workbook.save("source_vs_rag.xlsx")

    # Delete database
    db.client.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    evaluate.logging.set_verbosity_info()
    evaluate.enable_progress_bar()
    main()
