import asr
from qg import qg
import warnings
from qg import QGPipeline
from chunking import ChunkPipeline
from filtering import slide_chunking, get_similarity_scores, filter_questions_by_retention_rate



def baseline(
    video_path: str, slides_path: str | None, similarity_threshold, filtering_threshold
) -> list[str]:
    qg_model = qg.Model.DISCORD
    asr_model = asr.Model.DISTIL_SMALL
    asr_pipe = asr.ASRPipeline(asr_model)
    whisper_chunks, duration = asr_pipe(video_path, batch_size=1)
    chunk_pipe = ChunkPipeline(qg_model)
    transcript_chunks = chunk_pipe(whisper_chunks, duration)

    text_chunks = [chunk["text"] for chunk in transcript_chunks]
    qg_pipe = QGPipeline(qg_model)
    generated_questions = qg_pipe(text_chunks)

    if slides_path is None:
        return generated_questions

    slide_chunks = slide_chunking(video_path, slides_path)
    sim_scores = get_similarity_scores(duration, transcript_chunks, slide_chunks)
    filtered_questions = filter_questions_by_retention_rate(sim_scores, generated_questions, similarity_threshold, filtering_threshold)

    if not filtered_questions:
        warnings.warn(
            "Could not effectively perform question filtering, all generated questions are being returned"
        )
        return generated_questions
    else:
        return filtered_questions

if __name__ == "__main__":
    import argparse
    import pprint
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video file path")
    parser.add_argument("slides", nargs="?", help="slides file path", default=None)
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        help="threshold for slides filtering",
        default=0.5,
    )
    parser.add_argument(
        "--filtering_threshold",
        type=float,
        help="threshold for percentage of filtered questions",
        default=0.5,
    )
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()

    if args.verbose:
        transformers.logging.set_verbosity(transformers.logging.DEBUG)

    questions = baseline(
        args.video, args.slides, args.similarity_threshold, args.filtering_threshold
    )

    pprint.pprint(questions)
