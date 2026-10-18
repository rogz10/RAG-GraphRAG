from pathlib import Path

import mlflow
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from eval.corpus import load_eval_chunks
from eval.evaluate import MAX_SEQ_LENGTH, build_evaluator, data_dir, load_split

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # le modèle B
OUTPUT_DIR = Path("models/rag-embeddings-fr")
SEED = 42

def build_train_dataset(rows: list[dict], corpus: dict[str, str]) -> Dataset:
    """paires (question, bon passage) pour l'entraînement"""
    return Dataset.from_dict({
        "anchor": [row["question"] for row in rows],
        "positive": [corpus[row["chunk_id"]] for row in rows],
    })

if __name__ == "__main__":
    chunks, _ = load_eval_chunks(data_dir)
    corpus = {chunk.metadata["chunk_id"]: chunk.page_content for chunk in chunks}
    train_dataset = build_train_dataset(load_split("train"), corpus)
    evaluator = build_evaluator(load_split("val"), corpus, "val")

    model = SentenceTransformer(BASE_MODEL,model_kwargs={"attn_implementation":"eager"})
    model.max_seq_length = MAX_SEQ_LENGTH
    loss = CachedMultipleNegativesRankingLoss(model,mini_batch_size=8)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_cosine_accuracy@4",
        seed=SEED,
        logging_steps=5,
        report_to=["mlflow"],
        run_name="C-finetune-val",
    )
    mlflow.set_experiment("evaluate_models-rag-retrieval")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    trainer.model.save_pretrained(str(OUTPUT_DIR / "final"))
    print("modèle sauvegardé dans", OUTPUT_DIR / "final")
