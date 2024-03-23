import csv
import random
from alinet import asr, qg, chunking, Question
from alinet.chunking.similarity import (
    filter_questions_by_retention_rate,
    get_similarity_scores,
    filter_similar_questions,
)
from alinet.chunking.video import save_video_clips, slide_chunking
import warnings

from alinet.rag.db import Database
from chromadb import Collection


def baseline(
    video_path: str,
    asr_model: asr.Model,
    qg_model: qg.Model,
    pdfs_bytes: list[bytes],
) -> list[Question]:
    asr_pipe = asr.Pipeline(asr_model)
    whisper_chunks, duration = asr_pipe(video_path, batch_size=1)
    chunk_pipe = chunking.Pipeline(qg_model)
    transcript_chunks = chunk_pipe(whisper_chunks, duration)

    if len(pdfs_bytes) != 0:
        # Supplementary material
        # Get database singleton instance
        db = Database._instance
        collection: Collection = db.create_collection()
        db.store_documents(collection, pdfs_bytes=pdfs_bytes)
        text_chunks = [
            db.add_relevant_context_to_source(context=chunk.text, collection=collection)
            for chunk in transcript_chunks
        ]
    else:
        text_chunks = [chunk.text for chunk in transcript_chunks]

    qg_pipe = qg.Pipeline(qg_model)
    generated_questions = qg_pipe(text_chunks)
    filtered_questions = filter_similar_questions(generated_questions)

    slide_chunks = slide_chunking(video_path)
    if len(slide_chunks) == 0:
        warnings.warn(
            "Slide chunks are empty. Questions will not have a similarity score."
        )
        return [
            Question(text=question, similarity_score=0.0)
            for question in filtered_questions
        ]

    sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)

    return [
        Question(text=question, similarity_score=score)
        for question, score in zip(generated_questions, sim_scores)
    ]


def create_eval_questions(
    video_path: str,
    output_dir_path: str,
    asr_model: asr.Model,
    qg_model: qg.Model,
    similarity_threshold: float,
    filtering_threshold: float,
    stride_time: int,
    sample_size: int,
    seed: int,
):
    asr_pipe = asr.Pipeline(asr_model)
    whisper_chunks, duration = asr_pipe(video_path, batch_size=1)
    chunk_pipe = chunking.Pipeline(qg_model)
    transcript_chunks = chunk_pipe(whisper_chunks, duration, stride_length=32)

    text_chunks = [chunk.text for chunk in transcript_chunks]
    qg_pipe = qg.Pipeline(qg_model)
    generated_questions = qg_pipe(text_chunks)

    slide_chunks = slide_chunking(video_path)
    sim_scores = get_similarity_scores(transcript_chunks, slide_chunks, overlap=15)
    assert len(transcript_chunks) == len(sim_scores)
    filtered_questions = filter_questions_by_retention_rate(
        sim_scores,
        generated_questions,
        similarity_threshold,
        filtering_threshold,
    )

    random.seed(seed)
    keys_list = random.sample(list(filtered_questions.keys()), sample_size)

    with open(f"{output_dir_path}/questions.csv", "w", newline="") as file:
        writer = csv.writer(file)
        field = ["index", "chunk", "question", "similarity"]

        writer.writerow(field)

        for (idx, question), chunk, sim in zip(
            enumerate(generated_questions), transcript_chunks, sim_scores
        ):
            writer.writerow([idx, chunk.text, question, sim])

    save_video_clips(
        video_path, transcript_chunks, output_dir_path, keys_list, stride_time
    )
