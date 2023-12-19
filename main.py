import asr
import qg
from asr import ASRPipeline
from qg import QGPipeline
from chunking import ChunkPipeline
from utils import compute_similarity_between_source


def baseline(video_path: str, slides_path: str | None, threshold) -> list[str]:
    qg_model = qg.Model.DISCORD
    asr_model = asr.Model.DISTIL_SMALL
    asr_pipe = ASRPipeline(asr_model)
    whisper_chunks, duration = asr_pipe(video_path, batch_size=1)
    chunk_pipe = ChunkPipeline(qg_model)
    chunks = chunk_pipe(whisper_chunks, duration)

    text_chunks = [chunk["text"] for chunk in chunks]
    qg_pipe = QGPipeline(qg_model)
    generated_questions = qg_pipe(text_chunks)

    if slides_path is None:
        return generated_questions

    sim_scores = compute_similarity_between_source(text_chunks, slides_path)
    return [
        question
        for sim, question in zip(sim_scores, generated_questions)
        if sim > threshold
    ]


if __name__ == "__main__":
    import argparse
    import pprint
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video file path")
    parser.add_argument("slides", nargs="?", help="slides file path", default=None)
    parser.add_argument(
        "--threshold", type=float, help="threshold for slides filtering", default=0.5
    )
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()

    if args.verbose:
        transformers.logging.set_verbosity(transformers.logging.DEBUG)

    questions = baseline(args.video, args.slides, args.threshold)

    pprint.pprint(questions)
