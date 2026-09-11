from fastapi import FastAPI
from agent import run_agent
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=TMP_DIR), name="outputs")


class ChatRequest(BaseModel):
    query: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat")
async def chat(payload: ChatRequest):
    response = run_agent(payload.query)
    return {"response": response}
