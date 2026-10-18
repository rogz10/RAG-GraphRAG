from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings

SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}


def load_documents(data_dir: Path | None = None) -> list[Document]:
    """charge les documents depuis le dossier source"""
    source_dir = data_dir or settings.data_raw_dir
    if not source_dir.exists():
        return []

    documents: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            documents.append(_load_text(path))

    return documents


def _load_text(path: Path) -> Document:
    """charge un document texte"""
    return Document(
        page_content=path.read_text(encoding="utf-8"),
        metadata={"source": str(path)},
    )


def _load_pdf(path: Path) -> list[Document]:
    """charge chaque page utile d un document pdf"""
    documents = []
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        content = page.extract_text()
        if content and content.strip():
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "page": page_number,
                        "source": str(path),
                    },
                ),
            )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """découpe les documents en morceaux exploitables par le rag"""
    splitter = RecursiveCharacterTextSplitter(
        add_start_index=True,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return splitter.split_documents(documents)
