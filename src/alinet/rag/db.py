from transformers import BartTokenizer, HfArgumentParser
import chromadb
from chromadb.api import ClientAPI
from chromadb import Collection
from chromadb.config import Settings

from angle_emb import AnglE
import logging
from spacy.lang.en import English

from alinet.qg import Model

import hashlib
import fitz
import io

from dataclasses import dataclass, field
import torch


@dataclass
class RAGDatabaseArguments:
    context: str = field(metadata={"help": "Context to add relevant information to"})
    doc_paths: list[str] = field(
        metadata={"help": "List of document paths"},
    )
    top_k: int = field(
        default=1,
        metadata={"help": "Number of relevant contexts to retrieve"},
    )
    distance_threshold: float = field(
        default=0.5,
        metadata={"help": "Distance threshold to consider a context as relevant"},
    )


# Helper function
def generate_sha256_hash_from_text(text):
    # Create a SHA256 hash object
    sha256_hash = hashlib.sha256()
    # Update the hash object with the text encoded to bytes
    sha256_hash.update(text.encode("utf-8"))
    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()


logger = logging.getLogger(__name__)
nlp = English()
nlp.add_pipe("sentencizer")


class Database:
    def __init__(
        self,
        pretrained_bart_tokenizer_name: Model = Model.BALANCED_RESOLVED,
        output_dir: str = "./chromadb",
    ):
        # Tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(pretrained_bart_tokenizer_name)

        self.angle: AnglE

        if torch.cuda.is_available():
            self.angle = AnglE.from_pretrained(
                "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
            ).cuda()
        else:
            self.angle = AnglE.from_pretrained(
                "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
            )

        # ChromaDB client and collection
        settings = Settings()
        settings.allow_reset = True
        self.client: ClientAPI = chromadb.PersistentClient(
            path=output_dir, settings=settings
        )

    def _get_doc_text(self, pdf_bytes: bytes):
        texts = []
        with io.BytesIO(pdf_bytes) as pdf_stream:
            doc = fitz.open(stream=pdf_stream)
            for page in doc:
                texts.append(page.get_text())
        return "".join(texts)

    # Splits a document into chunks of text, aiming to respect a maximum token limit.
    def _create_document_chunks(self, document: str, chunk_size: int = 512):
        doc = nlp(document)
        sentences = [span.text for span in doc.sents]
        tokenized_sents = self.tokenizer(sentences)

        token_i = 0
        doc_i = 0
        documents = [[]]
        for sent, tokens in zip(sentences, tokenized_sents["input_ids"]):
            if token_i + len(tokens) >= chunk_size:
                token_i = 0
                documents.append([])
                doc_i += 1
            documents[doc_i].append(sent)
            token_i += len(tokens)

        return [" ".join(d) for d in documents]

    def store_documents(
        self,
        collection: Collection,
        pdfs_bytes: list[bytes],
        max_token_limit: int = 512,
    ):
        for pdf_bytes in pdfs_bytes:
            document = self._get_doc_text(pdf_bytes)
            chunks = self._create_document_chunks(document, max_token_limit)

            for doc in chunks:
                embedding = self.angle.encode(doc, to_numpy=True)

                collection.add(
                    embeddings=embedding,
                    documents=doc,
                    ids=generate_sha256_hash_from_text(doc),
                )

    # Makes sure that an old collection with the same name is deleted, so that a new one is created
    def create_collection(self, collection_name: str = "default"):
        try:
            self.client.delete_collection(collection_name)
        except ValueError:
            logger.info(f"{collection_name} does not exist")

        collection: Collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # l2 is the default
        )

        return collection

    def add_relevant_context_to_source(
        self,
        context: str,
        collection: Collection,
        distance_threshold: float = 0.5,
        top_k: int = 1,
    ):
        query_embedding = self.angle.encode(context, to_numpy=True)

        relevant_queries = collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )

        for i in range(top_k):
            if relevant_queries["distances"][i][0] > distance_threshold:
                relevant_context = relevant_queries["documents"][i][0]

                context = f"{context} {relevant_context}"

        return context


if __name__ == "__main__":
    parser = HfArgumentParser((RAGDatabaseArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    db = Database()
    collection = db.create_collection()

    pdfs_bytes: list[bytes] = []
    for doc_path in args.doc_paths:
        with open(doc_path, "rb") as f:
            pdf_bytes = f.read()
        pdfs_bytes.append(pdf_bytes)

    db.store_documents(collection, pdfs_bytes=pdfs_bytes, max_token_limit=32)

    result = db.add_relevant_context_to_source(
        args.context, collection, args.distance_threshold, args.top_k
    )
    print(result)
    db.client.reset()
