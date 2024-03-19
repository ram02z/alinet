import os
import sys
from dataclasses import dataclass, field
import pprint

import transformers
from transformers import HfArgumentParser

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)

from alinet import baseline, qg, asr  # noqa: E402


@dataclass
class BaselineArguments:
    video: str = field(metadata={"help": "Video file path"})
    similarity_threshold: float = field(
        default=0.5, metadata={"help": "Threshold for slides filtering"}
    )
    filtering_threshold: float = field(
        default=0.5, metadata={"help": "Threshold for percentage of filtered questions"}
    )
    qg_model: qg.Model = field(
        default=qg.Model.BALANCED_RA,
        metadata={"help": "Question generation model to use"},
    )
    video_clips_path: str | None = field(
        default=None,
        metadata={"help": "Directory to save the video clips"},
    )
    asr_model: asr.Model = field(
        default=asr.Model.DISTIL_LARGE,
        metadata={"help": "Automatic Speech Recongition model to use"},
    )
    doc_paths: list[str] = field(
        default_factory=list,
        metadata={"help": "List of document paths. Add paths to documents separated by spaces"},
    )
    verbose: bool = field(default=False, metadata={"help": "Increase output verbosity"})


if __name__ == "__main__":
    parser = HfArgumentParser((BaselineArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    if args.verbose:
        transformers.logging.set_verbosity(transformers.logging.DEBUG)

    questions = baseline(
        args.doc_paths,
        args.video,
        args.similarity_threshold,
        args.filtering_threshold,
        args.asr_model,
        args.qg_model,
        args.video_clips_path
    )

    pprint.pprint(questions)
