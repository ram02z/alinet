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
    texts: list[str] = field(metadata={"help": "Texts to add relevant information to"})
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
        elif torch.backends.mps.is_available():
            mps_device = torch.device("mps")
            self.angle = AnglE.from_pretrained(
                "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
            ).to(mps_device)
        else:
            self.angle = AnglE.from_pretrained(
                "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
            )

        # ChromaDB client and collection
        settings = Settings()
        settings.allow_reset = True
        settings.anonymized_telemetry = False
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
    def _create_document_chunks(self, document: str, chunk_size: int):
        doc = nlp(document)
        sentences = [span.text for span in doc.sents]

        if not sentences:
            return []

        tokenized_sents = self.tokenizer(sentences)
        documents = []
        current_document = []
        current_length = 0

        # Initialize the first document with the first sentence
        # Otherwise, we might have an empty first document
        current_document = [sentences[0]]
        current_length = len(tokenized_sents["input_ids"][0])

        for sent, tokens in zip(sentences[1:], tokenized_sents["input_ids"][1:]):
            if current_length + len(tokens) >= chunk_size:
                documents.append(" ".join(current_document))
                current_document = []
                current_length = 0
            current_document.append(sent)
            current_length += len(tokens)

        if current_document:
            documents.append(" ".join(current_document))

        return documents

    def store_documents(
        self,
        collection: Collection,
        pdfs_bytes: list[bytes],
        max_token_limit: int = 32,
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

    def add_relevant_context_to_sources(
        self,
        source_texts: list[str],
        collection: Collection,
        distance_threshold: float = 0.5,
        top_k: int = 1,
    ):
        query_embeddings = self.angle.encode(source_texts, to_numpy=True)

        query_result = collection.query(
            query_embeddings=query_embeddings, n_results=top_k
        )

        sources_with_context = []
        for i, source_text in enumerate(source_texts):
            context = []
            for j in range(len(query_result["distances"][i])):
                if query_result["distances"][i][j] > distance_threshold:
                    document = query_result["documents"][i][j]
                    context.append(document)

            source_with_context = " ".join([source_text, *context])
            sources_with_context.append(source_with_context)

        return sources_with_context


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

    db.store_documents(collection, pdfs_bytes=pdfs_bytes)

    result = db.add_relevant_context_to_sources(
        args.texts, collection, args.distance_threshold, args.top_k
    )
    print(result)
    db.client.reset()
