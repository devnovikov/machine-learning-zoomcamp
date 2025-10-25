import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI()

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

def predict_single(datapoint):
    result = pipeline.predict_proba(datapoint)[0, 1]
    return float(result)

@app.post("/predict")
def predict(client: Client):
    probability = predict_single(client.model_dump())
    return {"subscription_probability": float(probability)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)