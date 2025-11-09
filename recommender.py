import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

MODEL_NAME = "all-MiniLM-L6-v2"

def _normalize_url(u: str) -> str:
    if not isinstance(u, str): return ""
    u = u.strip().lower()
    # strip scheme + host, keep path
    for prefix in ["https://www.shl.com", "http://www.shl.com", "https://shl.com", "http://shl.com"]:
        if u.startswith(prefix):
            u = u[len(prefix):]
    # normalize /solutions/products/... -> /products/...
    u = u.replace("/solutions/products/", "/products/")
    # dedupe slashes and trim trailing slash
    if u.endswith("/"):
        u = u[:-1]
    return u

class SHLRecommender:
    def __init__(self, csv_path="assessments.csv", use_reranker: bool=True, pool:int=50):
        # 1) embedder
        self.model = SentenceTransformer(MODEL_NAME)

        # 2) catalog (your real scraped CSV)
        self.df = pd.read_csv(csv_path)

        # normalize column names from your friend's scrape
        self.df.rename(columns={"job_title":"name","job_url":"url"}, inplace=True)

        # description used for embeddings
        if "test_types" in self.df.columns:
            self.df["description"] = self.df["name"].astype(str) + ". Test types: " + self.df["test_types"].astype(str)
        else:
            self.df["description"] = self.df["name"].astype(str)

        # pre-compute embeddings
        self.embeddings = self.model.encode(
            self.df["description"].tolist(),
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # 3) optional reranker
        self.use_reranker = use_reranker
        self.pool = pool
        self.reranker = None
        if use_reranker and Path("reranker-model").exists():
            self.reranker = CrossEncoder("reranker-model")

        # 4) precompute canonical URL for matching
        self.df["canon_url"] = self.df["url"].map(_normalize_url)

    def recommend(self, query: str, top_k: int = 10):
        # Stage 1: embedding retrieval (top-N pool)
        q_emb = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(q_emb, self.embeddings)[0]
        k_pool = min(self.pool, len(self.df))
        top_idx = scores.topk(k_pool).indices.cpu().numpy()
        cand = self.df.iloc[top_idx].copy()

        # Stage 2: rerank with cross-encoder if available
        if self.reranker is not None:
            pairs = [(query, f'{row["name"]} — {row["description"]}') for _, row in cand.iterrows()]
            re_scores = self.reranker.predict(pairs)  # higher is better
            cand["re_score"] = re_scores
            cand = cand.sort_values("re_score", ascending=False)

        # Final top-k
        k = min(top_k, len(cand))
        return cand.head(k)[["name","url","canon_url"]]
    
    # === EXPORTED FUNCTION USED BY app.py ===

_recommender = SHLRecommender()

def get_recommendations(query: str):
    results = _recommender.recommend(query, top_k=10)
    return results["url"].tolist()