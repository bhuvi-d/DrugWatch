from fastapi import FastAPI
from pydantic import BaseModel

# In the final project, these will be imported like:
#   from extract import run_extraction
#   from classify import classify_sentence
# For now, we define temporary placeholder functions.

def run_extraction(text: str):
    """
    Mock version of NER pipeline.
    Pretends to extract drug-ADE pairs from text.
    """
    return [
        {"drug": "aspirin", "event": "nausea"},
        {"drug": "ibuprofen", "event": "headache"}
    ]

def classify_sentence(text: str):
    """
    Mock version of ADE classifier.
    Pretends to classify a sentence as ADE or Non-ADE.
    """
    if "pain" in text.lower() or "nausea" in text.lower():
        return "ADE", 0.85
    else:
        return "Non-ADE", 0.92

# -------------------------------
# FastAPI app setup 
# -------------------------------

app = FastAPI(title="DrugWatch Backend", version="0.1")

class TextInput(BaseModel):
    """Input schema for /extract and /classify endpoints."""
    text: str

@app.get("/healthcheck")
def healthcheck():
    """
    Simple healthcheck endpoint.
    Used to verify that the backend is running.
    """
    return {"status": "ok"}

@app.post("/extract")
def extract_entities(input: TextInput):
    """
    endpoint:
    Calls extraction pipeline (mocked for now).
    Input: { "text": "..." }
    Output: { "extractions": [ {drug, event}, ... ] }
    """
    results = run_extraction(input.text)
    return {"extractions": results}

@app.post("/classify")
def classify_text(input: TextInput):
    """
    endpoint:
    Calls classifier (mocked for now).
    Input: { "text": "..." }
    Output: { "label": "ADE/Non-ADE", "confidence": float }
    """
    label, confidence = classify_sentence(input.text)
    return {"label": label, "confidence": confidence}
