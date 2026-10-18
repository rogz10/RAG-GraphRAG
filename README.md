# RAG sur la réglementation énergétique des zones non interconnectées

Assistant de questions-réponses sur des documents publics de la CRE et de l'État (délibérations, programmations pluriannuelles de l'énergie). Il concerne les zones non interconnectées : La Réunion, Martinique, Guyane.

Le projet contient une API RAG, un protocole d'évaluation de la recherche de passages et un fine-tuning du modèle d'embeddings.

## Ce qui est fait

- Pipeline RAG : chargement des PDF, découpage en chunks, base vectorielle Chroma, réponse par un LLM local avec Ollama.
- API FastAPI (`/health`, `/ingest`, `/ask`, `/metrics`) et image Docker.
- Jeu d'évaluation de 714 questions générées par `gemma3:27b`, découpé en train / validation / test par passage.
- Comparaison de deux modèles d'embeddings et de BM25, suivie dans MLflow.
- Fine-tuning du modèle multilingue sur les questions d'entraînement.
- Ontologie du domaine : 7 classes, 11 relations.
- Prototype d'extraction de triplets par LLM, vérifiés par l'ontologie.
- CI GitHub Actions : ruff, pytest, build Docker.

## Ce qui reste à faire

- Mesurer les modèles sur le jeu de test.
- Lancer l'extraction de triplets sur tout le corpus et mesurer sa précision.
- Construire le graphe de connaissances et la recherche multi-sauts.
- Comparer les modes vectoriel, graphe et hybride.

## Résultats

Recherche du bon passage parmi 386, sur les 66 questions de validation. hit@4 est la métrique principale, car le RAG envoie 4 passages au LLM.

| modèle | hit@1 | hit@4 | hit@5 | MRR@10 |
|---|---|---|---|---|
| A : all-MiniLM-L6-v2 | 0,091 | 0,167 | 0,197 | 0,140 |
| BM25 | 0,061 | 0,227 | 0,273 | 0,162 |
| B : paraphrase-multilingual-MiniLM-L12-v2 | 0,152 | 0,288 | 0,348 | 0,226 |
| C : B après fine-tuning | 0,303 | 0,515 | 0,530 | 0,413 |

Le fine-tuning fait passer hit@4 de 0,288 à 0,515. Il utilise `MultipleNegativesRankingLoss` sur 504 paires question / passage, pendant 3 époques.

Une première version des questions, générée avec `llama3.2`, recopiait trop le texte : BM25 trouvait le bon passage en premier dans 80 % des cas. Elles ont été régénérées avec `gemma3:27b`, puis filtrées pour retirer les questions trop proches du texte ou trop vagues.

## Outils

- Python, FastAPI, LangChain, Chroma, Ollama
- sentence-transformers, PyTorch, rank_bm25, MLflow
- Pydantic, prometheus-client
- pytest, ruff, Docker, GitHub Actions

Matériel :
- génération des questions sur un GPU NVIDIA L40S (RunPod) ;
- fine-tuning sur un MacBook Pro M4

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2
```

## Utilisation

```bash
uvicorn app.main:app --reload
curl -X POST localhost:8000/ingest
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Quelle est la puissance installée de la centrale du Gol ?"}'
```

Avec Docker :

```bash
docker compose up --build
```

Évaluation et fine-tuning :

```bash
pip install -r requirements-train.txt
python -m eval.evaluate
python -m eval.finetune
```

Les PDF du corpus d'évaluation sont listés dans `eval/corpus_source.md` et vont dans `data/corpus_eval/`.

Tests :

```bash
ruff check app tests eval
pytest -q
```

## Structure

```text
app/
  config.py              paramètres lus depuis .env
  ingestion.py           lecture des PDF et découpage en chunks
  vector_store.py        embeddings et base Chroma
  rag_pipeline.py        recherche des passages et réponse du LLM
  main.py                routes de l'API
  metrics.py             compteurs Prometheus
  ontologie.py           classes, relations et vérification d'un triplet
  extraction.py          extraction de triplets par LLM
eval/
  corpus_source.md       liste des documents et leurs liens
  corpus.py              chargement du corpus d'évaluation
  generate_questions.py  génération et filtrage des questions
  split.py               découpage train / validation / test
  review.py              échantillon de questions à relire
  evaluate.py            comparaison des modèles et de BM25
  finetune.py            fine-tuning du modèle d'embeddings
data/
  raw/                   PDF utilisés par l'API
  corpus_eval/           questions et découpage (fichiers .jsonl)
tests/                   tests unitaires
.github/workflows/       CI
Dockerfile, compose.yaml image et lancement de l'API
```

## Limites

- Les questions sont synthétiques.
- La validation contient 66 questions : les petits écarts ne sont pas significatifs.
- Le jeu de test n'a pas encore été mesuré.

## Auteur

Isidore Zongo
