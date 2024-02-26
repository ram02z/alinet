from alinet import asr, qg, chunking
from alinet.chunking.similarity import (
    get_similarity_scores,
    filter_questions_by_retention_rate,
)
from alinet.chunking.video import (
    slide_chunking, 
    save_video_clips
)

def baseline(
    video_path: str,
    similarity_threshold,
    filtering_threshold,
    asr_model,
    qg_model,
) -> list[str]:
    asr_pipe = asr.Pipeline(asr_model)
    whisper_chunks, duration = asr_pipe(video_path, batch_size=1)
    chunk_pipe = chunking.Pipeline(qg_model)
    transcript_chunks = chunk_pipe(whisper_chunks, duration)

    # Uncomment the following line to save video clips locally from each chunk
    # save_video_clips(video_path, transcript_chunks, output_dir="saved_clips")

    text_chunks = [chunk["text"] for chunk in transcript_chunks]
    qg_pipe = qg.Pipeline(qg_model)
    generated_questions = qg_pipe(text_chunks)

    slide_chunks = slide_chunking(video_path)
    sim_scores = get_similarity_scores(duration, transcript_chunks, slide_chunks)
    filtered_questions = filter_questions_by_retention_rate(
        sim_scores, generated_questions, similarity_threshold, filtering_threshold
    )

    return filtered_questions
