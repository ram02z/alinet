import qg
from qg import QGPipeline
from asr import ASRPipeline
from chunking import ChunkPipeline


def baseline(video_path: str, slides_path: str | None) -> list[str]:
    model = qg.Model.DISCORD
    asr_pipe = ASRPipeline()
    whisper_chunks, duration = asr_pipe(video_path, batch_size=1)
    chunk_pipe = ChunkPipeline(model)
    if slides_path:
        with open(args.slides, "rb") as slides:
            filtered_chunks = chunk_pipe(
                whisper_chunks, duration, pdf_stream=slides.read()
            )
    else:
        filtered_chunks = chunk_pipe(whisper_chunks, duration)

    text_chunks = [chunk["text"] for chunk in filtered_chunks]
    qg_pipe = QGPipeline(model)

    return qg_pipe(text_chunks)


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video file path")
    parser.add_argument("slides", nargs="?", help="slides file path", default=None)
    args = parser.parse_args()

    questions = baseline(args.video, args.slides)

    pprint.pprint(questions)
