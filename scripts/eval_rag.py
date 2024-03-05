from dataclasses import dataclass, field
import logging
import unicodedata
import datasets
from transformers import (
    HfArgumentParser,
    set_seed,
    BartForConditionalGeneration,
    BartTokenizer,
)
from datasets import load_dataset
from angle_emb import AnglE
import chromadb
from evaluate import Text2TextGenerationEvaluator
import evaluate

logger = logging.getLogger(__name__)


@dataclass
class EvaluateModelArguments:
    pretrained_bart_model_name: str = field(
        metadata={"help": "The name of the pretrained model"},
    )
    max_length: int = field(
        default=32,
        metadata={"help": "The maximum total input sequence length after tokenization"},
    )
    min_length: int = field(
        default=0,
        metadata={"help": "The minimum total input sequence length after tokenization"},
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})
    vectordb_path: str = field(
        default="./chromadb", metadata={"help": "The path to the vectordb"}
    )
    collection_name: str = field(
        default="pubmedqa_validation",
        metadata={"help": "The name of the collection in the vectordb"},
    )
    num_beams: int = field(
        default=4,
        metadata={"help": "Number of beams to use for evaluation"},
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})


def contain_question_mark(data):
    return data["target"][-1].rstrip() == "?"


def normalise(data):
    # Remove new line characters
    data["source"] = data["source"].replace("\n", " ")

    # Resolve accented characters
    data["source"] = "".join(
        c
        for c in unicodedata.normalize("NFD", data["source"])
        if unicodedata.category(c) != "Mn"
    )
    data["target"] = "".join(
        c
        for c in unicodedata.normalize("NFD", data["target"])
        if unicodedata.category(c) != "Mn"
    )

    return data


def main():
    parser = HfArgumentParser((EvaluateModelArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    logger.info("loading embedding model")
    angle = AnglE.from_pretrained(
        "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
    ).cuda()

    logger.info("loading vectordb")
    client = chromadb.PersistentClient(path=args.vectordb_path)
    collection = client.get_collection(args.collection_name)

    def add_relevant_context_to_source(data):
        query_embedding = angle.encode(data["source"], to_numpy=True)

        relevant_query = collection.query(query_embeddings=query_embedding, n_results=1)

        relevant_context = relevant_query["documents"][0][0]

        long_answer_with_relevant_context = f"{data['source']} {relevant_context}"

        data["source"] = long_answer_with_relevant_context

        return data

    logger.info("loading dataset")
    eval_data = (
        load_dataset("alinet/pubmed_qa", "truncated_512", split="validation")
        .select_columns(["source", "target"])
        .map(add_relevant_context_to_source)
        .filter(contain_question_mark)
        .map(normalise)
    )

    logger.info("loading model")
    model = BartForConditionalGeneration.from_pretrained(
        args.pretrained_bart_model_name
    )
    tokenizer = BartTokenizer.from_pretrained(args.pretrained_bart_model_name)

    logger.info("loading metric")
    metric = evaluate.load("bertscore")

    logger.info("evaluating model")
    evaluator = Text2TextGenerationEvaluator()
    evaluator.METRIC_KWARGS = {"model_type": "microsoft/deberta-xlarge-mnli"}
    results = evaluator.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=eval_data,
        metric=metric,
        input_column="source",
        label_column="target",
        random_state=args.seed,
        generation_kwargs={
            "max_new_tokens": args.max_length,
            "min_new_tokens": args.min_length,
            "num_beams": args.num_beams,
        },
    )

    logger.info("saving results")
    params = {
        "model": args.pretrained_bart_model_name,
    }
    evaluate.save("./results/", **results, **params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    evaluate.logging.set_verbosity_info()
    evaluate.enable_progress_bar()
    main()
