from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400


@patch("app.main.load_documents", return_value=[])
def test_ingest_rejects_empty_directory(load_documents):
    response = client.post("/ingest")

    assert response.status_code == 400
    load_documents.assert_called_once()


@patch("app.main.index_documents")
@patch("app.main.split_documents")
@patch("app.main.load_documents")
def test_ingest_indexes_documents(load_documents, split_documents, index_documents):
    documents = [Document(page_content="Document de test")]
    chunks = [Document(page_content="Chunk de test")]
    load_documents.return_value = documents
    split_documents.return_value = chunks

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json() == {"documents": 1, "chunks": 1}
    index_documents.assert_called_once_with(chunks)


@patch("app.main.ask_question")
def test_ask_returns_rag_response(ask_question):
    ask_question.return_value = {
        "answer": "Réponse de test",
        "sources": [],
    }

    response = client.post("/ask", json={"question": "Quelle est la réponse"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Réponse de test"


def test_metrics_counts_requests():
    client.get("/health")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert 'rag_requests_total{endpoint="/health",status="200"}' in response.text
