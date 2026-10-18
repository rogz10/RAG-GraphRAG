import json
import re
from pathlib import Path

import mlflow
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)

from eval.corpus import load_eval_chunks

data_dir = Path("data/corpus_eval") # dossier contenant les documents d'évaluation
MAX_SEQ_LENGTH = 512  # longueur maximale des séquences pour le modèle de langage
#models
MODELS={
    "A":"sentence-transformers/all-MiniLM-L6-v2", # modèle de langage pour l'anglais
    "B":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # modèle de langage pour le multilingue
}

# charger ensemble de questions
def load_split(name:str)->list[dict]:
    """
    charge les questions depuis le fichier JSONL correspondant à l'ensemble spécifié
    retourne une liste de dictionnaires représentant les questions
    """
    path=data_dir/f"{name}.jsonl"
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f] # charger les questions depuis le fichier JSONL
    print(f"{len(rows)} questions chargées depuis {path}")
        #line = f.readline()
        # afficher quelques lignes pour vérifier le contenu du fichier
    for row in rows[:3]:
        print(row)
    return rows

def build_evaluator(rows: list[dict],corpus: dict[str,str], name:str) -> InformationRetrievalEvaluator:
    """
    construit un évaluateur d'information retrieval à partir des questions et du corpus
    retourne un objet InformationRetrievalEvaluator
    préparer l'examen : question , corpus entuer et bonnes réponses
    """
    queries={str(i): row["question"] for i, row in enumerate(rows)} # dictionnaire des questions avec l'index comme clé
    relevant_docs={str(i): [row["chunk_id"]] for i, row in enumerate(rows)} # dictionnaire id question et ensemble des bon chunks_id

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        accuracy_at_k=[1,4,5], # évaluer la précision à k=1,4,5 hit@k
        mrr_at_k=[10], # évaluer le mean reciprocal rank à k=10 mrr@k
        show_progress_bar=True, # afficher la barre de progression
    )
def bm25_metrics(rows: list[dict], corpus: dict[str, str]) -> dict[str, float]:
    """mesure la recherche par mots communs (BM25), référence sans réseau de neurones"""
    ids = list(corpus)
    tokenize = lambda text: re.findall(r"\w+", text.lower())
    bm25 = BM25Okapi([tokenize(corpus[i]) for i in ids])
    hit_1 = hit_4 = hit_5 = mrr = 0.0
    for row in rows:
        scores = bm25.get_scores(tokenize(row["question"]))
        ranking = [ids[k] for k in sorted(range(len(ids)), key=lambda k: -scores[k])]
        rank = ranking.index(row["chunk_id"]) + 1  # rang du bon chunk (1 = premier)
        hit_1 += rank == 1
        hit_4 += rank <= 4
        hit_5 += rank <= 5
        mrr += 1 / rank if rank <= 10 else 0
    n = len(rows)
    return {"hit_at_1": hit_1 / n, "hit_at_4": hit_4 / n, "hit_at_5": hit_5 / n, "mrr_at_10": mrr / n}

if __name__ == "__main__":
    chunks, _ = load_eval_chunks(data_dir)
    corpus = {chunk.metadata["chunk_id"]: chunk.page_content for chunk in chunks}
    val_rows = load_split("val")
    evaluator = build_evaluator(val_rows, corpus, "val") # construire l'évaluateur pour l'ensemble de validation
    mlflow.set_experiment("evaluate_models-rag-retrieval") # définir l'expérience mlflow
    for label, model_name in MODELS.items():
        model = SentenceTransformer(model_name,model_kwargs={"attn_implementation":"eager"})
        model.max_seq_length = MAX_SEQ_LENGTH
        print(f"Évaluation du modèle {label} : {model_name}")
        with mlflow.start_run(run_name=f"{label}-val"):
            mlflow.log_params({
                "model": model_name,
                "max_seq_length": MAX_SEQ_LENGTH,
                "split": "val",
                "n_questions": len(val_rows),
                "n_corpus": len(corpus),
            })
            results = evaluator(model)
            metrics = {
                "hit_at_1": results["val_cosine_accuracy@1"],
                "hit_at_4": results["val_cosine_accuracy@4"],
                "hit_at_5": results["val_cosine_accuracy@5"],
                "mrr_at_10": results["val_cosine_mrr@10"],
            }
            mlflow.log_metrics(metrics)
        print(label, {key: round(value, 3) for key, value in metrics.items()})

    # référence BM25
    with mlflow.start_run(run_name="BM25-val"):
        mlflow.log_params({"model": "BM25", "split": "val", "n_questions": len(val_rows), "n_corpus": len(corpus)})
        bm25_results = bm25_metrics(val_rows, corpus)
        mlflow.log_metrics(bm25_results)
    print("BM25", {key: round(value, 3) for key, value in bm25_results.items()})

