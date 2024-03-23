import logging
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import sys
import os
import tempfile
from pydantic import BaseModel

SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)
from alinet import asr, qg, baseline, Question  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionsResponse(BaseModel):
    questions: list[Question]


@app.post("/generate_questions", response_model=QuestionsResponse)
async def generate_questions(files: List[UploadFile] = File(...)):
    videos = [file for file in files if file.content_type == "video/mp4"]
    if not videos:
        raise HTTPException(status_code=400, detail="No video files provided")

    temp_video_paths = []
    try:
        for video in videos:
            file_type = os.path.splitext(video.filename)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_type
            ) as temp_file:
                shutil.copyfileobj(video.file, temp_file)
                logger.info(f"file '{video.filename}' saved as '{temp_file.name}'")
                temp_video_paths.append(temp_file.name)
    except Exception as e:
        cleanup_files(temp_video_paths)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing files: {str(e)}",
        )

    pdfs_bytes = [
        await file.read() for file in files if file.content_type == "application/pdf"
    ]

    questions = []
    for temp_video_path in temp_video_paths:
        generated_questions = baseline(
            video_path=temp_video_path,
            asr_model=asr.Model.DISTIL_MEDIUM,
            qg_model=qg.Model.BALANCED_RA,
            pdfs_bytes=pdfs_bytes,
        )
        questions.extend(generated_questions)

    cleanup_files(temp_video_paths)

    return QuestionsResponse(questions=questions)


def cleanup_files(temp_file_paths):
    for temp_file_path in temp_file_paths:
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
