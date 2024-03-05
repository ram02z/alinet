import datasets
from transformers import HfArgumentParser, BartTokenizer, set_seed
from dataclasses import dataclass, field
import chromadb
from angle_emb import AnglE
from datasets import load_dataset
import logging
from tqdm import tqdm
from spacy.lang.en import English

logger = logging.getLogger(__name__)
nlp = English()
nlp.add_pipe("sentencizer")


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


# Splits a document into chunks of text, aiming to respect a maximum token limit.
def create_documents(document, tokenizer, chunk_size):
    doc = nlp(document)
    sentences = [span.text for span in doc.sents]
    tokenized_sents = tokenizer(sentences)

    token_i = 0
    doc_i = 0
    documents = [[]]
    for sent, tokens in zip(sentences, tokenized_sents['input_ids']):
        if token_i + len(tokens) >= chunk_size:
            token_i = 0
            documents.append([])
            doc_i += 1
        documents[doc_i].append(sent)
        token_i += len(tokens)

    return [' '.join(d) for d in documents]

def main():
    parser = HfArgumentParser((GenerateArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    # Tokenizer
    tokenizer = BartTokenizer.from_pretrained(args.pretrained_bart_tokenizer_name)

    # ChromaDB client and collection
    client = chromadb.PersistentClient(path=args.output_dir)
    collection = create_collection(client, args.collection_name)

    # Embedding Model
    angle = AnglE.from_pretrained(
        "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
    ).cuda()

    pubmed_ds = load_dataset("alinet/pubmed_qa", "truncated_512", split="validation")

    for data in tqdm(pubmed_ds):
        documents = create_documents(data["context"], tokenizer, args.max_token_limit)

        for id, doc in enumerate(documents):
            embedding = angle.encode(doc, to_numpy=True)

            chunk_id = f"{data['pubid']}C{id}"
            collection.add(
                embeddings=embedding,
                documents=doc,
                ids=chunk_id,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    datasets.logging.set_verbosity_info()
    main()
