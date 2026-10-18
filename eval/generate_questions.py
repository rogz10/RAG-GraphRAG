import json
import re
import time
from pathlib import Path

from langchain_ollama import ChatOllama
from pydantic import BaseModel

from eval.corpus import load_eval_chunks

output_path = Path("data/corpus_eval/questions.jsonl")
min_questions_length = 15  # longueur minimale des questions générées

#generator_model="qwen2.5:7b" # modèle de génération de questions
# model su gpu l40s
generator_model="gemma3:27b" # modèle de génération de questions
max_copy_ratio=0.5 # chevauchement maximal entre les questions générées
max_chunk=None # nombre maximal de chunks à traiter pour la génération de questions
vague_reference = re.compile(r"\b(cette|cet|ces|mentionnée?s?|ci-dessus|ce (rapport|document|texte|passage|type))\b")


class GeneratedQuestion(BaseModel):
    """classe pour représenter une question générée"""
    #questions:list[str]# liste de questions générées
    question_1: str
    question_2: str


    # prompt
PROMPT = """Tu écris des questions pour tester un moteur de recherche documentaire.
Lis le passage, puis écris deux questions auxquelles ce passage répond.

Règle principale : la question est posée par quelqu'un qui n'a PAS lu le passage et qui utilise SES propres mots.
- N'utilise pas les expressions techniques du passage : remplace-les par des mots courants ou des synonymes.
- Pas de sigles, pas de numéros d'article, pas de citation du passage.
- Chaque question doit se suffire à elle-même : nomme le sujet précis (lieu, dispositif, année) avec tes mots. N'écris jamais « cette », « ces », « ce rapport », « l'organisme mentionné ».
- Une seule phrase par question, sans la réponse. Ne dis pas « ce passage » ou « ce document ».

Exemple :
Passage : « Le tarif réduit est accordé aux voyageurs de moins de 26 ans. »
Mauvaise question (recopie) : Quel tarif réduit est accordé aux voyageurs de moins de 26 ans ?
Bonne question (reformulée) : Jusqu'à quel âge peut-on payer son billet moins cher ?

Passage :
{passage}"""
def copy_ratio(question: str, passage: str) -> float:
    """part des mots de la question (4 lettres ou plus) recopiés du passage"""
    question_words = {w for w in re.findall(r"\w+", question.lower()) if len(w) >= 4}
    passage_words = {w for w in re.findall(r"\w+", passage.lower()) if len(w) >= 4}
    if not question_words:
        return 0.0
    return len(question_words & passage_words) / len(question_words)


def generate_questions(llm, text: str) -> list[str]:
        """ demader au llm 2  questions reformulées à partir d'un passage de texte"""
        result = llm.invoke(PROMPT.format(passage=text))
        return [result.question_1, result.question_2 ]

if __name__ == "__main__":
    # charger les docs
    chunks, _ = load_eval_chunks(Path("data/corpus_eval"))
    chunks = chunks[:max_chunk]  # limiter le nombre de chunks à traiter
    # initialiser le llm
    llm = ChatOllama(model=generator_model, temperature=0, num_predict=200).with_structured_output(GeneratedQuestion)# formulaire de sortie structurée

    seen_questions = set()# ensemble des questions déjà générées
    rejected_questions = {"llm_error": 0, "too_short": 0, "duplicate": 0, "too_lexical": 0,"vague_reference": 0} # dictionnaire des questions rejetées
    kept=0# nombre de questions conservées
    start_time = time.time()# temps de départ
    with output_path.open("w", encoding="utf-8") as f:
         for i, chunk in enumerate(chunks,start=1):
            try:
                questions = generate_questions(llm, chunk.page_content)
            except Exception as e: # noqa: BLE001 (toute erreur du LLM est comptée, puis on continue)
                print(f"Erreur lors de la génération des questions pour le chunk {i}: {e}")
                rejected_questions["llm_error"] += 1
                continue

            for question in questions:
                question = question.strip()  # enlever les espaces
                if len(question) < min_questions_length: # si la question est trop courte, on la rejette
                    rejected_questions["too_short"] += 1
                    continue
                if question in seen_questions: # si la question est déjà vue, on la rejette
                    rejected_questions["duplicate"] += 1
                    continue
                if vague_reference.search(question): # si la question contient une référence vague, on la rejette
                    rejected_questions["vague_reference"] += 1
                    continue
                ratio = copy_ratio(question, chunk.page_content) # calculer le ratio de recopie
                # test
                #print(f"ratio={ratio:.0%} question={question}")
                if ratio > max_copy_ratio: # si le ratio de recopie est trop élevé, on la rejette
                    rejected_questions["too_lexical"] += 1
                    continue
                seen_questions.add(question) #sinon on ajoute la question à l'ensemble des questions vues
                #la ligne à écrire dans le fichier de sortie
                row={
                    "question": question,
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "source": chunk.metadata.get("source"),
                    "page": chunk.metadata.get("page"),
                }
                #écrire la ligne dans le fichier de sortie
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1 # incrémenter compteur de questions conservées

            # Afficher la progression toutes les 20 itérations
            if i % 20 == 0:
                print(f"Chunks{i}/{len(chunks)}, Temps écoulé: {time.time() - start_time:.2f}s")
    duration=time.time() - start_time
    print(f"question gardées:{kept}")
    print(f"question rejetées:{rejected_questions}")
    print(f"Temps total écoulé: {duration:.2f}s, Temps par chunk: {duration/len(chunks):.2f}s")
