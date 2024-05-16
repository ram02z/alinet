import logging
from chromadb import Collection
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import sys
import os
import tempfile
from pydantic import BaseModel
from contextlib import asynccontextmanager
from alinet.chunking.model import TimeChunk
from alinet.chunking.video import slide_chunking
from alinet.model import Document, TextWithReferences
import itertools

from alinet.rag import Database

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)
from alinet import asr, qg, rag, baseline, Question  # noqa: E402

logger = logging.getLogger(__name__)

db: Database | None = None
collection: Collection | None = None
video_path: str | None = None
slide_chunks: list[TimeChunk] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    global collection
    db = rag.Database()
    collection = db.create_collection()
    yield
    if not db.client.reset():
        logger.warning("database collections and entries could not be deleted")
    del db


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionsResponse(BaseModel):
    questions: list[Question]


@app.post("/generate_questions", response_model=QuestionsResponse)
async def generate_questions(
    top_k: int = Form(...),
    distance_threshold: float = Form(...),
):
    if not video_path:
        raise HTTPException(status_code=400, detail="No video file uploaded")

    logger.info(
        f"RAG parameters: top_k = {top_k}, distance_threshold = {distance_threshold}"
    )
    def query_collection(source_texts: list[str]) -> list[TextWithReferences]:
        if collection:
            return db.add_relevant_context_to_sources(
                source_texts=source_texts,
                collection=collection,
                top_k=top_k,
                distance_threshold=distance_threshold,
            )
        else:
            return [TextWithReferences(text=text, ref=None) for text in source_texts]

    questions = []
    generated_questions = baseline(
        video_path=video_path,
        asr_model=asr.Model.DISTIL_MEDIUM,
        qg_model=qg.Model.BALANCED_RA,
        augment_sources=query_collection,
        slide_chunks=slide_chunks
    )
    questions.extend(generated_questions)

    return QuestionsResponse(questions=questions)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".mp4"}
    file_name, file_type = os.path.splitext(file.filename)

    if file_type.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type",
        )

    if file_type.lower() == ".pdf":
        pdf = Document(name=file.filename, content=await file.read())
        db.store_documents(collection, docs=[pdf])
    elif file_type.lower() == ".mp4":
        try:
            tempfile._get_candidate_names = lambda: itertools.repeat(file_name)
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_type, prefix=""
            ) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                logger.info(f"file '{file.filename}' saved as '{temp_file.name}'")
                global video_path
                video_path = temp_file.name
                global slide_chunks
                slide_chunks = slide_chunking(video_path)
                cleanup_files(video_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while processing files: {str(e)}",
            )


def cleanup_files(temp_file_path):
    try:
        os.remove(temp_file_path)
        logger.info(f"file '{temp_file_path}' removed")
    except Exception as e:
        logger.error(f"error removing file '{temp_file_path}': {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=["web_app"],
    )
