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
    doc_paths: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of document paths. Add paths to documents separated by spaces"
        },
    )
    qg_model: qg.Model = field(
        default=qg.Model.BALANCED_RA,
        metadata={"help": "Question generation model to use"},
    )
    asr_model: asr.Model = field(
        default=asr.Model.DISTIL_LARGE,
        metadata={"help": "Automatic Speech Recognition model to use"},
    )
    verbose: bool = field(default=False, metadata={"help": "Increase output verbosity"})


if __name__ == "__main__":
    parser = HfArgumentParser((BaselineArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    if args.verbose:
        transformers.logging.set_verbosity(transformers.logging.DEBUG)

    pdfs_bytes: list[bytes] = []
    for doc_path in args.doc_paths:
        with open(doc_path, "rb") as f:
            pdf_bytes = f.read()
        pdfs_bytes.append(pdf_bytes)

    questions = baseline(
        args.video,
        asr_model=args.asr_model,
        qg_model=args.qg_model,
        pdfs_bytes=pdfs_bytes,
    )

    pprint.pprint(questions)
