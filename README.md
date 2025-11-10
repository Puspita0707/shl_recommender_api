🚀 SHL Assessment Recommendation System

Intelligent LLM-Enhanced Assessment Recommendation API:

This repository contains my complete SHL assessment recommendation system, built according to all requirements in the assignment:

✔ Crawling + preprocessing SHL catalog

✔ Embedding retrieval using SentenceTransformers

✔ Fine-tuned Cross-Encoder reranking

✔ Balanced recommendations (Technical + Behavioral)

✔ Evaluation using Recall@10

✔ FastAPI backend

✔ Deployed on HuggingFace Spaces

✔ Submission CSV generator

✔ Clean GitHub repository for submission

🔗 Live Services

🔹 Backend API (FastAPI, HuggingFace Space):

👉 https://pushpa12234-shl-recommender-hf.hf.space

🔹 HuggingFace Code Repository (Backend + UI):

👉 https://huggingface.co/spaces/Pushpa12234/SHL_RECOMMENDER_UI/tree/main

🔹 GitHub Repository (Submission Repo):

👉 https://github.com/Puspita0707/shl_recommender_api

🔹 HuggingFace UI:( Run this to get the frontend)

👉 https://huggingface.co/spaces/Pushpa12234/SHL_RECOMMENDER_UI


1. System Architecture:
 
 Query → Embedding Retrieval (MiniLM)
       → Top-50 candidates
       → Cross-Encoder Reranker (fine-tuned)
       → Top-10 assessments
       → JSON API Output

This two-stage pipeline ensures high accuracy, semantic relevance, and balanced coverage across skill domains.

📌 2. Features:

🧠 Semantic Matching

Uses all-MiniLM-L6-v2 to embed both queries and assessment descriptions.

🔥 Reranking

Fine-tuned Cross-Encoder on labelled dataset improves accuracy.

⚖ Balanced Results

If query mentions behavioral + technical, both types appear.

📊 Full Evaluation

Metrics computed on validation/test set:

Recall@10:    0.224

Precision@10: 0.130

MAP@10:       0.390

NDCG@10:      0.469

📌 3. Repository Structure

/
├── app.py                         # FastAPI endpoint: /health, /recommend

├── recommender.py                 # Embeddings + reranking logic

├── assessments.csv                # Scraped catalog data

├── reranker-model/                # Fine-tuned CrossEncoder model

├── requirements.txt               # Dependencies

├── start.sh                       # Entrypoint for deployment

└── README.md                      # Documentation



📌 3. API DOCUMENTATION

/heath

GET /health

response: 

{ "status": "ok" }

/recommend

POST /recommend

request:

{
  "query": "Looking for a Python developer with strong teamwork"
}

response:

{
  "query": "Looking for a Python developer with strong teamwork",
  "recommendations": [
    {
      "assessment_name": "Computer Science (New)",
      "assessment_url": "https://www.shl.com/products/product-catalog/view/computer-science-new/"
    },
    ...
  ]
}


📌 5. Data Pipeline

✔ Catalog Scraping:

The product catalog was scraped and normalized into assessments.csv.

✔ Embedding Generation:

Used MiniLM encoder with normalized embeddings.

✔ Reranker Training:

Fine-tuned Cross-Encoder using the provided training set of labeled queries.

✔ Evaluation:

Metrics computed using official ranking measures required in assignment.

📌 6. Deployment (HuggingFace SPACES)

The system is deployed using:

FastAPI

Uvicorn

HuggingFace Spaces (Docker)

Everything is packaged so that anyone can run:

uvicorn app:app --host 0.0.0.0 --port 7860


📌 7. Submission CSV

format_submission.py produces CSV in exact format:

| Query   | Assessment_url |
| ------- | -------------- |
| Query 1 | https://...    |
| Query 1 | https://...    |
| Query 2 | https://...    |

📌 8. Tech Stack

FastAPI (backend)

Uvicorn (server)

Gradio (UI)

SentenceTransformers

Torch

Pandas

📌 9. Run Locally:

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860












