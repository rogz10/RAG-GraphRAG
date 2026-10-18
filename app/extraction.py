# librairies
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.config import settings
from app.ingestion import load_documents, split_documents
from app.ontologie import CLASSES, RELATIONS, verifier_triplet
from app.vector_store import _document_id

# listes fermées pour typer les classes et relations
ClassName = Literal[tuple(sorted(CLASSES))]
RelationName = Literal[tuple(RELATIONS)]

class Triplet(BaseModel):
    """un fait proposé par le llm"""

    sujet: str
    classe_sujet: ClassName
    relation: RelationName
    objet: str
    classe_objet: ClassName


class Extraction(BaseModel):
    """réponse attendue du llm : une liste de faits"""

    triplets: list[Triplet]
PROMPT = """Tu extrais des faits d'un texte réglementaire sur l'énergie.

Classes autorisées : {classes}

Relations autorisées (classe de départ -> classe d'arrivée) :
{relations}

Consignes :
- n'extrais que les faits écrits explicitement dans le texte, aucune déduction ;
- garde les noms tels qu'ils sont écrits dans le texte ;
- un nom ne contient jamais sa classe ;
- un territoire est une île ou une région (La Réunion, Martinique, Guyane), jamais une ville ni une direction ;
- chaque fait une seule fois ;
- si aucun fait ne correspond aux relations, rends une liste vide.

Exemple (fictif : n'en reprends jamais les noms)
Texte : « La société Énergie Sud, filiale du groupe Volta, exploite la centrale de Port-Bleu, située en Guadeloupe, qui fonctionne au fioul. »
Faits :
- sujet=Énergie Sud, classe_sujet=Societe, relation=FILIALE_DE, objet=Volta, classe_objet=Societe
- sujet=Énergie Sud, classe_sujet=Societe, relation=EXPLOITE, objet=Port-Bleu, classe_objet=Centrale
- sujet=Port-Bleu, classe_sujet=Centrale, relation=SITUEE_A, objet=Guadeloupe, classe_objet=Territoire
- sujet=Port-Bleu, classe_sujet=Centrale, relation=UTILISE, objet=fioul, classe_objet=Combustible

Texte :
{chunk}
"""

def build_prompt(text: str) -> str:
    """construire le prompt à partir de l'ontologie et du chunk"""
    relations = "\n".join(f"- {name} : {start} -> {end}" for name, (start, end) in RELATIONS.items())
    return PROMPT.format(classes=", ".join(sorted(CLASSES)), relations=relations, chunk=text) # retourne le prompt complet à envoyer au llm

def call_llm(prompt: str) -> Extraction:
    """demande les faits au llm au format du schéma"""
    llm = ChatOllama(model=settings.extraction_model, temperature=0)
    return llm.with_structured_output(Extraction, method="json_schema").invoke(prompt)


def extract_chunk(chunk: Document) -> tuple[list[dict], list[dict]]:
    """extraire les faits d'un chunk et les trie avec l'ontologie"""
    accepted, rejected = [], []
    for triplet in call_llm(build_prompt(chunk.page_content)).triplets:
        row = {
            **triplet.model_dump(),
            "chunk_id": _document_id(chunk),
            "source": Path(chunk.metadata["source"]).name,
            "page": chunk.metadata.get("page"),
        }
        errors = verifier_triplet(triplet.classe_sujet, triplet.relation, triplet.classe_objet)
        if errors:
            rejected.append({**row, "erreurs": errors})
        else:
            accepted.append(row)
    return accepted, rejected

def is_skipped(chunk: Document) -> bool:
    """tableaux horaires de la cre 2024-65 (pages 7 à 13) : rien pour l'ontologie"""
    return "cre_2024-65" in chunk.metadata["source"] and chunk.metadata.get("page", 0) >= 7


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    chunks = [chunk for chunk in split_documents(load_documents()) if not is_skipped(chunk)][:limit]
    out_dir = settings.data_processed_dir
    reasons = Counter()
    n_accepted = n_rejected = 0
    with (
        open(out_dir / "triplets.jsonl", "w", encoding="utf-8") as accepted_file,
        open(out_dir / "triplets_rejetes.jsonl", "w", encoding="utf-8") as rejected_file,
    ):
        for index, chunk in enumerate(chunks, start=1):
            accepted, rejected = extract_chunk(chunk)
            accepted_file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted)
            rejected_file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected)
            for row in rejected:
                reasons.update(row["erreurs"])
            n_accepted += len(accepted)
            n_rejected += len(rejected)
            print(f"{index}/{len(chunks)} : {len(accepted)} acceptés, {len(rejected)} rejetés")

    print(f"total : {n_accepted} acceptés, {n_rejected} rejetés")
    for reason, count in reasons.most_common():
        print(f"{count:4d}  {reason}")
