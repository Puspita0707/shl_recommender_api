from fastapi import FastAPI
from pydantic import BaseModel
from recommender import get_recommendations

app = FastAPI()

class Query(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(q: Query):
    results = get_recommendations(q.query)
    # Must return at most 10
    results = results[:10]

    return {
        "query": q.query,
        "recommendations": [
            {"assessment_url": url} for url in results
        ]
    }
