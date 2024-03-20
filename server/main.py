from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import sys
import os
import tempfile
from typing import List

SRC_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(SRC_DIR)
from alinet import baseline
from alinet import asr, qg

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
    for file in files:
        if file.content_type not in ["video/mp4"]:
            raise Exception(f"File type not allowed: {file.filename}")

    temp_file_paths = []
    for file in files:
        try:
            file_type = os.path.splitext(file.filename)[1]

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_type
            ) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_paths.append(temp_file.name)
        except Exception as e:
            print(e)
            return {"message": "There was an error processing the file"}
    print(temp_file_paths)

    questions = []
    for temp_file_path in temp_file_paths:
        questions_for_temp_file = baseline(
            temp_file_path,
            similarity_threshold=0.5,
            filtering_threshold=0.5,
            asr_model=asr.Model.DISTIL_MEDIUM,
            qg_model=qg.Model.BALANCED_RA,
        )
        for key, value in questions_for_temp_file.items():
            questions.append(value)

    return {"questions": questions}
