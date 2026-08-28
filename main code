from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Classifier API")

# CORS: allows your frontend (running on a different origin/port) to call this API.
# Without this, browsers block the request by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production, replace "*" with your actual frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, reused for every request (loading it per-request would be slow).
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
# Swap in your own fine-tuned model once you've trained one:
# classifier = pipeline("text-classification", model="your-username/imdb-distilbert-sentiment")


class TextInput(BaseModel):
    text: str


class Prediction(BaseModel):
    label: str
    confidence: float


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Sentiment classifier API is running"}


@app.post("/classify", response_model=Prediction)
def classify(input: TextInput):
    """
    Takes {"text": "some review"} and returns {"label": "POSITIVE", "confidence": 0.98}
    """
    result = classifier(input.text)[0]
    return Prediction(label=result["label"], confidence=round(result["score"], 4))


# Run locally with: uvicorn main:app --reload --port 8000
