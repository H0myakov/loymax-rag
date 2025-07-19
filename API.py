from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import query_chromadb

app = FastAPI(title="RAG API with ChromaDB + Ollama")

# Модель входа
class QueryRequest(BaseModel):
    question: str

# Модель ответа
class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks: list[str]


@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    result = query_chromadb(req.question)
    return QueryResponse(
        question=result["query"],
        answer=result["answer"],
        chunks=result["chunks"]
    )
