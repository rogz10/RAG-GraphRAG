from unittest.mock import patch

from langchain_core.documents import Document

from app.extraction import Extraction, Triplet, build_prompt, extract_chunk
from app.ontologie import RELATIONS


def test_build_prompt_contains_ontology_and_text():
    prompt = build_prompt("texte du chunk")

    assert "texte du chunk" in prompt
    for relation in RELATIONS:
        assert relation in prompt


@patch("app.extraction.call_llm")
def test_extract_chunk_sorts_with_ontology(call_llm):
    call_llm.return_value = Extraction(
        triplets=[
            Triplet(sujet="ABR", classe_sujet="Centrale", relation="SITUEE_A", objet="La Réunion", classe_objet="Territoire"),
            Triplet(sujet="ABR1", classe_sujet="Unite", relation="UTILISE", objet="charbon", classe_objet="Combustible"),
        ],
    )
    chunk = Document(page_content="texte", metadata={"source": "data/raw/doc.pdf", "page": 2})

    accepted, rejected = extract_chunk(chunk)

    assert [row["sujet"] for row in accepted] == ["ABR"]
    assert [row["sujet"] for row in rejected] == ["ABR1"]
    assert accepted[0]["source"] == "doc.pdf"
    assert accepted[0]["page"] == 2
