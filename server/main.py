
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
import shutil
import sys
import os
import tempfile
from typing import List

SRC_DIR = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(SRC_DIR)
from src.alinet.main import baseline


app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware( 
    CORSMiddleware,
    allow_origins=origins,
)

@app.post("/generate_questions")
async def generate_questions(files: List[UploadFile] = File(...)):  
    for file in files:
        if file.content_type not in ["video/mp4", "application/pdf"]:
            raise Exception(f"File type not allowed: {file.filename}")
        
    temp_file_paths = []
    for file in files:
        try:
            file_type = os.path.splitext(file.filename)[1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_paths.append(temp_file.name)
        except Exception as e:
            print(e)
            return {"message": "There was an error processing the file"}
    print(temp_file_paths)
    
    questions = []
    for temp_file_path in temp_file_paths:  
        questions_for_temp_file = baseline(temp_file_path, 0.5, 0.5, "distil-whisper/distil-medium.en", "alinet/bart-base-squad-qg", None)
        questions = questions + questions_for_temp_file

    return {"questions": questions}


 