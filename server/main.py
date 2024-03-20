import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import sys
import os
import tempfile
from typing import List

SRC_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(SRC_DIR)
from alinet import asr, qg, baseline  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate_questions")
async def generate_questions(files: List[UploadFile] = File(...)):
    videos = [file for file in files if file.content_type == "video/mp4"]
    if not videos:
        raise HTTPException(status_code=400, detail="No video files provided")

    temp_file_paths = []
    try:
        for video in videos:
            file_type = os.path.splitext(video.filename)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_type
            ) as temp_file:
                shutil.copyfileobj(video.file, temp_file)
                logger.info(f"file '{video.filename}' saved as '{temp_file.name}'")
                temp_file_paths.append(temp_file.name)
    except Exception as e:
        cleanup_files(temp_file_paths)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing files: {str(e)}",
        )

    questions = []
    for temp_file_path in temp_file_paths:
        generated_questions = baseline(
            temp_file_path,
            similarity_threshold=0.5,
            filtering_threshold=0.5,
            asr_model=asr.Model.DISTIL_MEDIUM,
            qg_model=qg.Model.BALANCED_RA,
        )
        questions.extend(generated_questions.values())

    cleanup_files(temp_file_paths)

    return {"questions": questions}


def cleanup_files(temp_file_paths):
    for temp_file_path in temp_file_paths:
        try:
            os.remove(temp_file_path)
            logger.info(f"file '{temp_file_path}' removed")
        except Exception as e:
            logger.error(f"error removing file '{temp_file_path}': {e}")
