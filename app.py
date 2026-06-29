from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("scam_model.pkl")


class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
def predict(req: TextRequest):

    prediction = model.predict([req.text])[0]

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba([req.text])[0][1])
    else:
        probability = None

    return {
        "text": req.text,
        "prediction": int(prediction),
        "label": "Scam" if prediction == 1 else "Safe",
        "confidence": probability
    }