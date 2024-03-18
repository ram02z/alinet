from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware( 
    CORSMiddleware,
    allow_origins=origins,
)

@app.get("/hello_world")
async def hello_world():
  return {"message": "Hello World"} 
 