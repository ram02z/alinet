from alinet import asr, qg, chunking
from alinet.chunking.similarity import (
    get_similarity_scores,
    filter_questions_by_retention_rate,
    filter_similar_questions,
)
from alinet.chunking.video import slide_chunking
import warnings
from alinet.rag.db import Database
from chromadb import Collection


def baseline(
    video_path: str,
    doc_paths: list[str],
    similarity_threshold,
    filtering_threshold,
    asr_model,
    qg_model,
) -> list[str]:
    asr_pipe = asr.Pipeline(asr_model)
    whisper_chunks, duration = asr_pipe(video_path, batch_size=1)
    chunk_pipe = chunking.Pipeline(qg_model)
    transcript_chunks = chunk_pipe(whisper_chunks, duration)

    # TODO: We might want to move this instantiate of the DB elsewhere, but I'm just gonna put it here for now
    if len(doc_paths) != 0:
        # Supplementary material
        db = Database()
        collection: Collection = db.create_collection(db.client)
        db.store_documents(collection, doc_paths=doc_paths)
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
            "Slide chunks are empty. Question filtering step is skipped. Non-filtered questions are returned"
        )
        return filtered_questions

    sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)
    filtered_questions = filter_questions_by_retention_rate(
        sim_scores, filtered_questions, similarity_threshold, filtering_threshold
    )

    return list(filtered_questions.values())
