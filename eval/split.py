import json
import random
from pathlib import Path

questions_path = Path("data/corpus_eval/questions.jsonl")
output_dir= Path("data/corpus_eval")

seed= 42  # valeur de départ pour la génération aléatoire

def split_questions(chunk_ids:list[str], seed: int = 42, train_ratio: float = 0.7, val_ratio: float = 0.1) -> dict[str, set[str]]:
    """
    divise les identifiants de chunks en ensembles d'entraînement, de validation et de test
    retourne trois listes d'identifiants de chunks
    """
    ids=sorted(set(chunk_ids))
    random.Random(seed).shuffle(ids) # mélange les identifiants de chunks de manière aléatoire

    n_train=int(len(ids) * train_ratio)# nombre d'identifiants de chunks pour l'ensemble d'entraînement
    n_val=int(len(ids) * val_ratio)# nombre d'identifiants de chunks pour


    return {
        "train": set(ids[:n_train]),
        "val": set(ids[n_train:n_train+n_val]),
        "test": set(ids[n_train+n_val:]),
    }


if __name__ == "__main__":
    # charger les questions generées
    with questions_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f] # charger les questions depuis le fichier JSONL
    split = split_questions([row["chunk_id"] for row in rows])# diviser les identifiants de chunks en ensembles d'entraînement, de validation et de test

    for name,ids in split.items():
        selected_rows = [row for row in rows if row["chunk_id"] in ids]# sélectionner les questions correspondant aux identifiants de chunks
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in selected_rows: # écrire les questions dans le fichier de sortie
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(selected_rows)} questions", f"({len(ids)} chunks)")
