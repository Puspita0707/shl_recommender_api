from fastapi import FastAPI
from pydantic import BaseModel
from SHL_assessment.SHL_API.recommender import SHLRecommender

app = FastAPI()
rec = SHLRecommender()

class QueryInput(BaseModel):
    text: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(input: QueryInput):
    results = rec.recommend(input.text, top_k=input.top_k)
    return results.to_dict(orient="records")