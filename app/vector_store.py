from hashlib import sha256

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


def get_embeddings():
    """crée le modèle qui transforme le texte en vecteurs"""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vector_store():
    """retourne la base vectorielle locale chromadb"""
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.vector_db_dir),
    )


def index_documents(documents: list[Document]):
    """ajoute les documents découpés dans la base vectorielle"""
    vector_store = get_vector_store()
    ids = [_document_id(document) for document in documents]
    vector_store.add_documents(documents, ids=ids)
    return vector_store


def _document_id(document: Document) -> str:
    """crée un identifiant stable pour éviter les doublons"""
    identity = "|".join(
        [
            str(document.metadata.get("source", "")),
            str(document.metadata.get("page", "")),
            document.page_content,
        ],
    )
    return sha256(identity.encode("utf-8")).hexdigest()
