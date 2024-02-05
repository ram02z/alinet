import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)

from alinet import baseline

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
