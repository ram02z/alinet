from alinet import asr, qg, chunking
from alinet.chunking.similarity import (
    get_similarity_scores,
    filter_questions_by_retention_rate,
)
from alinet.chunking.video import slide_chunking, save_video_clips
import warnings


def baseline(
    video_path: str,
    similarity_threshold,
    filtering_threshold,
    asr_model,
    qg_model,
    video_clips_path=None,
) -> dict[int, str]:
    asr_pipe = asr.Pipeline(asr_model)
    whisper_chunks, duration = asr_pipe(video_path, batch_size=1)
    chunk_pipe = chunking.Pipeline(qg_model)
    transcript_chunks = chunk_pipe(whisper_chunks, duration)

    if video_clips_path:
        save_video_clips(video_path, transcript_chunks, video_clips_path)

    text_chunks = [chunk.text for chunk in transcript_chunks]
    qg_pipe = qg.Pipeline(qg_model)
    generated_questions = qg_pipe(text_chunks)

    slide_chunks = slide_chunking(video_path)
    if len(slide_chunks) == 0:
        warnings.warn(
            "Slide chunks are empty. Question filtering step is skipped. Non-filtered questions are returned"
        )
        return {idx: question for idx, question in enumerate(generated_questions)}

    sim_scores = get_similarity_scores(transcript_chunks, slide_chunks)
    filtered_questions = filter_questions_by_retention_rate(
        sim_scores, generated_questions, similarity_threshold, filtering_threshold
    )

    return filtered_questions
