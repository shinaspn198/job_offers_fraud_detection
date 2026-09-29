from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model import predict
from .explain import word_impact, get_reasons
from .extract import extract_text
from .rag import rag_answer

app = FastAPI(title="Fraud Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # vite dev port
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str

class AskRequest(BaseModel):
    question: str
    document_text: str

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    verdict, fraud_score, legit_score = predict(req.text)
    reasons = get_reasons(req.text, verdict)
    impacts = word_impact(req.text)
    return {
        "verdict": verdict,
        "fraud_score": fraud_score,
        "legit_score": legit_score,
        "reasons": reasons,
        "word_impact": [{"word": w, "impact": d} for w, d in impacts],
    }

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text(content, file.content_type)
    return {"text": text}

@app.post("/ask")
def ask(req: AskRequest):
    answer = rag_answer(req.question, req.document_text)
    return {"answer": answer}

@app.get("/")
def health():
    return {"status": "ok"}