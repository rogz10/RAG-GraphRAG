import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# charge les variables du fichier env local
load_dotenv(BASE_DIR / ".env")


def _get_path(name: str, default: Path) -> Path:
    """retourne un chemin depuis une variable d environnement ou une valeur par défaut"""
    value = os.getenv(name)
    if not value:
        return default

    # si le chemin est relatif on le rattache à la racine du projet
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


@dataclass(frozen=True)
class Settings:
    """configuration centrale de l application rag"""

    # nom affiché par l application ou l api
    app_name: str = os.getenv("APP_NAME", "Personal RAG")

    # dossiers principaux utilisés pendant le traitement des documents
    data_raw_dir: Path = _get_path("DATA_RAW_DIR", BASE_DIR / "data" / "raw")
    data_processed_dir: Path = _get_path(
        "DATA_PROCESSED_DIR",
        BASE_DIR / "data" / "processed",
    )
    vector_db_dir: Path = _get_path("VECTOR_DB_DIR", BASE_DIR / "vector_db")

    # paramètres liés à la base vectorielle et aux modèles ia
    collection_name: str = os.getenv("COLLECTION_NAME", "personal_documents")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2")

    # paramètres du découpage documentaire et de la recherche sémantique
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000")) # taille maximale d un chunk en nombre de caractères
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150")) # chevauchement entre deux chunks consécutifs en nombre de caractères
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4")) # nombre de résultats de recherche à retourner


# instance partagée à importer dans les autres modules
settings = Settings()
