import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

BASE_MODEL_NAME = "google/flan-t5-small"
DATA_DIR = Path("data")
FEEDBACK_PATH = DATA_DIR / "user_feedback.jsonl"
SAVE_DIR = Path("models/finetuned_t5")

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128


class FeedbackQADataset(Dataset):
    def __init__(self, tokenizer, examples: List[dict]):
        self.tokenizer = tokenizer
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        context = "\n\n".join(ex["context_chunks"])
        question = ex["question"]
        target = ex["user_correct_answer"]

        prompt = (
            "You are a helpful study assistant for college notes.\n"
            "Use ONLY the given context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        model_inputs = self.tokenizer(
            prompt,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = self.tokenizer(
            target,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]

        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels.squeeze(0)

        for k in ["input_ids", "attention_mask"]:
            model_inputs[k] = model_inputs[k].squeeze(0)

        return model_inputs


def load_feedback_examples():
    examples = []
    if not FEEDBACK_PATH.exists():
        print("No feedback file found.")
        return examples

    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            user_ans = (obj.get("user_correct_answer") or "").strip()
            if not user_ans:
                continue
            if not obj.get("context_chunks"):
                continue
            examples.append(
                {
                    "question": obj["question"],
                    "user_correct_answer": user_ans,
                    "context_chunks": obj["context_chunks"],
                }
            )
    return examples


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading feedback examples...")
    examples = load_feedback_examples()
    print(f"Loaded {len(examples)} training examples.")
    if len(examples) == 0:
        print("No training data. Collect feedback first.")
        return

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME).to(device)

    dataset = FeedbackQADataset(tokenizer, examples)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(SAVE_DIR),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned model to {SAVE_DIR} ...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
