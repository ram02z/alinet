import os
import sys
from dataclasses import dataclass, field
import pprint

import transformers
from transformers import HfArgumentParser

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)

from alinet import baseline


@dataclass
class BaselineArguments:
    video: str = field(metadata={"help": "Video file path"})
    slides: str | None = field(default=None, metadata={"help": "Video file path"})
    similarity_threshold: float = field(
        default=0.5, metadata={"help": "Threshold for slides filtering"}
    )
    filtering_threshold: float = field(
        default=0.5, metadata={"help": "Threshold for percentage of filtered questions"}
    )
    verbose: bool = field(default=False, metadata={"help": "Increase output verbosity"})


if __name__ == "__main__":
    parser = HfArgumentParser((BaselineArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    if args.verbose:
        transformers.logging.set_verbosity(transformers.logging.DEBUG)

    questions = baseline(
        args.video, args.slides, args.similarity_threshold, args.filtering_threshold
    )

    pprint.pprint(questions)
