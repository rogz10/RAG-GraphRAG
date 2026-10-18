import time

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from app.config import settings
from app.ingestion import load_documents, split_documents
from app.metrics import REQUEST_DURATION, REQUESTS
from app.rag_pipeline import ask_question
from app.vector_store import index_documents

app = FastAPI(title=settings.app_name)

app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def track_requests(request: Request, call_next):
    """compte chaque requête et mesure sa durée"""
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")# route reconnue none si 404
    endpoint = route.path if route else "other"# url inconnue =other
    REQUESTS.labels(endpoint=endpoint, status=str(response.status_code)).inc()
    REQUEST_DURATION.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    return response


class AskRequest(BaseModel):
    """corps attendu pour poser une question"""

    question: str


@app.get("/health")
def health_check():
    """vérifie que l api répond"""
    return {"status": "ok"}


@app.post("/ingest")
def ingest_documents():
    """charge les documents et les indexe dans chromadb"""
    documents = load_documents()
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="Aucun document trouvé dans le dossier data raw",
        )

    chunks = split_documents(documents)
    index_documents(chunks)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
    }


@app.post("/ask")
def ask(request: AskRequest):
    """pose une question au système rag"""
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="La question ne peut pas être vide",
        )

    return ask_question(request.question)
