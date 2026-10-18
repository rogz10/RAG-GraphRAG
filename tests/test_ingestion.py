from pathlib import Path

from app.ingestion import load_documents, split_documents


def test_load_documents_reads_text_file(tmp_path):
    document_path = tmp_path / "document.txt"
    document_path.write_text("Contenu de test", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content == "Contenu de test"


def test_split_documents_creates_multiple_chunks():
    documents = load_documents(Path("tests/fixtures"))

    chunks = split_documents(documents * 20)

    assert len(chunks) >= 2
    assert all(chunk.page_content for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)
