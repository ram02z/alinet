import datasets
from transformers import HfArgumentParser, BartTokenizer, set_seed
from dataclasses import dataclass, field
from langchain.text_splitter import CharacterTextSplitter
import chromadb
from angle_emb import AnglE
from datasets import load_dataset
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class GenerateArguments:
    pretrained_bart_tokenizer_name: str = field(
        metadata={"help": "The name of the Bart Tokenizer model"},
    )
    max_token_limit: int = field(
        default=512,
        metadata={"help": "The max token limit for each chunk of the PMC article"},
    )
    output_dir: str = field(
        default="./chromadb", metadata={"help": "The output dir for the vectordb"}
    )
    collection_name: str = field(
        default="pubmedqa_validation",
        metadata={"help": "The name of the collection in vectordb"},
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})


# Makes sure that an old collection with the same name is deleted, so that a new one is created
def create_collection(client, collection_name):
    try:
        client.delete_collection(collection_name)
    except ValueError:
        logger.info(f"{collection_name} does not exist")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # l2 is the default
    )

    return collection


def main():
    parser = HfArgumentParser((GenerateArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    # Initialisations
    tokenizer = BartTokenizer.from_pretrained(args.pretrained_bart_tokenizer_name)

    # Text splitter
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=512, chunk_overlap=0
    )

    # ChromaDB client and collection
    client = chromadb.PersistentClient(path=args.output_dir)
    collection = create_collection(client, args.collection_name)

    # Embedding Model
    angle = AnglE.from_pretrained(
        "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
    ).cuda()

    pubmed_ds = load_dataset("alinet/pubmed_qa", "truncated_512", split="validation")

    for data in tqdm(pubmed_ds):
        documents = text_splitter.create_documents([data["context"]])

        for id, doc in enumerate(documents):
            embedding = angle.encode(doc.page_content, to_numpy=True)

            chunk_id = f"{data['pubid']}C{id}"
            collection.add(
                embeddings=embedding,
                documents=doc.page_content,
                ids=chunk_id,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
