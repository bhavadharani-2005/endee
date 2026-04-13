import json
import os
import requests
from pathlib import Path
from typing import Any
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
JOBS_FILE = BASE_DIR / "jobs.json"

ENDEE_URL = os.getenv("ENDEE_BASE_URL", "http://127.0.0.1:8080/api/v1")
INDEX_NAME = "jobs"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


# -----------------------------
# MODEL
# -----------------------------
def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# -----------------------------
# DATA
# -----------------------------
def load_jobs(file_path: Path = JOBS_FILE) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_job_text(job: dict[str, Any]) -> str:
    return f"{job.get('title','')} {job.get('description','')} {' '.join(job.get('skills', []))}"


# -----------------------------
# ENDEE REST FUNCTIONS
# -----------------------------
def create_index(dimension: int):
    url = f"{ENDEE_URL}/indexes"

    payload = {
        "name": INDEX_NAME,
        "dimension": dimension,
        "space_type": "cosine"
    }

    try:
        res = requests.post(url, json=payload)

        if res.status_code == 409:
            print("Index already exists")
        else:
            res.raise_for_status()
            print("Index created")

    except Exception as e:
        raise RuntimeError(f"Endee error: {e}")


def upsert_vectors(vectors):
    url = f"{ENDEE_URL}/indexes/{INDEX_NAME}/vectors"

    res = requests.post(url, json={"vectors": vectors})
    res.raise_for_status()


# -----------------------------
# MAIN INGEST
# -----------------------------
def ingest_jobs() -> int:
    jobs = load_jobs()
    model = get_embedder()

    dimension = model.get_sentence_embedding_dimension()

    # Create index safely
    create_index(dimension)

    payload = []

    for i, job in enumerate(jobs, start=1):
        text = build_job_text(job)

        vector = model.encode(text).tolist()

        payload.append({
            "id": str(i),
            "values": vector,
            "metadata": {
                "title": job.get("title", ""),
                "skills": job.get("skills", []),
                "description": job.get("description", "")
            }
        })

    upsert_vectors(payload)

    print(f"Inserted {len(payload)} jobs")

    return len(payload)


def ensure_jobs_index():
    return ingest_jobs()


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    ingest_jobs()
