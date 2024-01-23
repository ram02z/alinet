import asr
import qg
from asr import ASRPipeline
from qg import QGPipeline
from chunking import ChunkPipeline
from utils import compute_similarity_between_source
from chunk_filtering import get_similarity_scores


def baseline(video_path: str, slides_path: str | None, similarity_threshold, filtering_threshold) -> list[str]:
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

    sim_scores = get_similarity_scores(duration, text_chunks, video_path, slides_path)

    scores_and_questions = zip(sim_scores, generated_questions)
    filtered_questions = [question for sim, question in scores_and_questions if sim > similarity_threshold]
    filtering_percentage = len(filtered_questions) / len(generated_questions)

    return generated_questions if filtering_percentage < filtering_threshold else filtered_questions



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
