import os
import shutil
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import run_agent


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=TMP_DIR), name="outputs")

@app.get("/")
async def root():
    return {"message": "Hello World"}

SESSION_STORE = {
    "history": [],
    "images": []
}

@app.post("/chat")
async def chat(query: str = Form(...), images: List[UploadFile] = File(default=[])):
    global SESSION_STORE
    
    # Append any newly uploaded valid images
    if images and len(images) > 0:
        for image in images:
            if not image.filename:
                continue
            file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'png'
            unique_filename = f"{uuid.uuid4().hex[:8]}.{file_extension}"
            file_path = os.path.join(TMP_DIR, unique_filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            SESSION_STORE["images"].append(file_path)
            
        # Keep only the latest 2 images (since tools expect up to 2 images)
        SESSION_STORE["images"] = SESSION_STORE["images"][-2:]
        
    response = run_agent(query, image_paths=SESSION_STORE["images"], history=SESSION_STORE["history"])
    
    # Extract agent's text response for history
    agent_text = ""
    if isinstance(response, dict):
        if "analysis" in response:
            if isinstance(response["analysis"], dict):
                agent_text = response["analysis"].get("summary", "")
            else:
                agent_text = str(response["analysis"])
    else:
        agent_text = str(response)
        
    SESSION_STORE["history"].append({"user": query, "agent": agent_text})
    
    return {"response": response}
