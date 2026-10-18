from pathlib import Path

from langchain_core.documents import Document

from app.ingestion import load_documents, split_documents
from app.vector_store import _document_id


def letter_proportion(text:str) -> float:
    """
    calcule la proportion de lettres dans un texte et
    retourne la proportion de lettres dans le texte
    """
    if not text:
        return 0.0

    total_chars = len(text)
    letter_count = sum(1 for char in text if char.isalpha())

    return letter_count / total_chars

def usable_chunk(text:str,min_caracters:int=200,min_letters:float=0.5):
    """
    Vérifie si un chunk de texte est exploitable en fonction de sa longueur et de la proportion de lettres
    retourne True si le chunk est utilisable, False sinon
    """
    return len(text) >= min_caracters and letter_proportion(text) >= min_letters

def load_eval_chunks(data_dir:Path) -> tuple[list[Document], list[Document]]:
    """
    charge les documents d'évaluation depuis le dossier source
    retourne une liste de documents exploitables et une liste de documents non exploitables
    """
    if not data_dir.exists():
        return [], []
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    usable_docs=[]
    unusable_docs=[]
    for chunk in chunks:
        if usable_chunk(chunk.page_content):
            chunk.metadata["chunk_id"]=_document_id(chunk)
            usable_docs.append(chunk)
        else:
            unusable_docs.append(chunk)

    return usable_docs, unusable_docs


if __name__ == "__main__":

    data_dir = Path("data/corpus_eval")
    usable_docs, unusable_docs = load_eval_chunks(data_dir)
    print(f"Usable documents: {len(usable_docs)}")
    print(f"Unusable documents: {len(unusable_docs)}")
