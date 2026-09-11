from fastapi import FastAPI
from agent import run_agent
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    query: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat")
async def chat(payload: ChatRequest):
    response = run_agent(payload.query)
    return {"response": response}
