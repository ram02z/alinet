import os
import sys
from dataclasses import dataclass, field

import transformers
from transformers import HfArgumentParser

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)

from alinet import create_eval_questions, qg, asr  # noqa: E402


@dataclass
class CreateEvalQuestionsArguments:
    video: str = field(metadata={"help": "Video file path"})
    output_dir_path: str = field(
        metadata={"help": "Directory to save clips and questions"}
    )
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
    asr_model: asr.Model = field(
        default=asr.Model.DISTIL_LARGE,
        metadata={"help": "Automatic Speech Recognition model to use"},
    )
    stride_time: int = field(
        default=10, metadata={"help": "Time (in seconds) add to each clip"}
    )
    sample_size: int = field(
        default=10, metadata={"help": "Number of questions to sample"}
    )
    seed: int = field(default=1, metadata={"help": "Seed for random number generator"})
    verbose: bool = field(default=False, metadata={"help": "Increase output verbosity"})


if __name__ == "__main__":
    parser = HfArgumentParser((CreateEvalQuestionsArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    if args.verbose:
        transformers.logging.set_verbosity(transformers.logging.DEBUG)

    create_eval_questions(
        args.video,
        args.output_dir_path,
        args.asr_model,
        args.qg_model,
        args.similarity_threshold,
        args.filtering_threshold,
        args.stride_time,
        args.sample_size,
        args.seed,
    )
