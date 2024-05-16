import csv
import random
from alinet import asr, qg, chunking, Question
from alinet.chunking.model import TimeChunk
from alinet.chunking.similarity import (
    filter_questions_by_retention_rate,
    get_similarity_scores,
)
from alinet.chunking.video import save_video_clips, slide_chunking
from pathlib import Path
import warnings

from typing import Callable

from alinet.model import Reference, TextWithReferences


def baseline(
    video_path: str,
    asr_model: asr.Model,
    qg_model: qg.Model,
    slide_chunks: list[TimeChunk],
    augment_sources: Callable[[list[str]], list[TextWithReferences]],
) -> list[Question]:
    # asr_pipe = asr.Pipeline(asr_model)
    # whisper_chunks, duration = asr_pipe(video_path, batch_size=1)
    import json

    with open("./sample_data/lecture_5/demo/transcript.json", "rb") as f:
        data = json.load(f)
        whisper_chunks = data["chunks"]
        duration = data["duration"]
    chunk_pipe = chunking.Pipeline(qg_model)
    transcript_chunks = chunk_pipe(whisper_chunks, duration)

    reftext_chunks = augment_sources([chunk.text for chunk in transcript_chunks])
    text_chunks = [reftext.text for reftext in reftext_chunks]

    qg_pipe = qg.Pipeline(qg_model)
    generated_questions = qg_pipe(text_chunks)

    video_name = Path(video_path).name
    if len(slide_chunks) == 0:
        warnings.warn(
            "Slide chunks are empty. Questions will not have a similarity score."
        )
        questions = []
        for question, chunk, refs in zip(
            generated_questions, transcript_chunks, reftext_chunks
        ):
            chunk_ref = Reference(file_name=video_name, text=chunk.text)
            all_refs = []
            if refs.ref:
                all_refs = refs.ref
            all_refs.append(chunk_ref)
            questions.append(
                Question(text=question, similarity_score=0.0, refs=all_refs)
            )
        return questions

    sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)
    questions = []
    for question, chunk, refs, score in zip(
        generated_questions, transcript_chunks, reftext_chunks, sim_scores
    ):
        chunk_ref = Reference(file_name=video_name, text=chunk.text)
        all_refs = []
        if refs.ref:
            all_refs = refs.ref
        all_refs.append(chunk_ref)
        questions.append(Question(text=question, similarity_score=score, refs=all_refs))

    return questions


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
