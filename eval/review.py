import json
import random
from pathlib import Path

from eval.corpus import load_eval_chunks

questions_path = Path("data/corpus_eval/questions.jsonl")
review_path = Path("data/processed/review.md")

sample_size = 20 # nombre de questions à échantillonner pour la revue

if __name__ == "__main__":
    chunks, _ = load_eval_chunks(Path("data/corpus_eval"))
    text={chunk.metadata["chunk_id"]: chunk.page_content for chunk in chunks}# dictionnaire des passages de texte par identifiant de chunk
    with questions_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f] # charger les questions depuis le fichier JSONL
    sampled_rows = random.Random(42).sample(rows, (sample_size)) # échantillon
    with review_path.open("w", encoding="utf-8") as f:
        for n, row in enumerate(sampled_rows, start=1):
            f.write(f"## {n}. {row['question']}\n\n")
            f.write(f"- **Source**: {row['source']}\n")
            f.write(f"```\n{text[row['chunk_id']]}\n```\n\n")
            f.write("- [ ] la réponse est dans le passage\n")
            f.write("- [ ] la question est claire sans voir le passage\n\n")
    print(f"{sample_size} questions ecrite dans {review_path}")
