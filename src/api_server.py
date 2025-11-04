from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .retriever import Retriever


app = FastAPI(title="RAG Pipeline API")
_retriever: Retriever | None = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    retriever = get_retriever()
    results = retriever.search(req.query, top_k=req.top_k)
    response_items = [
        {"text": text, "score": score} for (text, score) in results
    ]
    return {"results": response_items}


# To run:
# uvicorn src.api_server:app --reload

