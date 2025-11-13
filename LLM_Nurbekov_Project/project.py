# ================================================
# Prompt Injection Classifier (clean version)
# - Model: protectai/deberta-v3-base-prompt-injection
# - Easy:  xTRam1/safe-guard-prompt-injection
# - Hard:  reshabhs/SPML_Chatbot_Prompt_Injection
# ================================================

import os, random
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import matplotlib.pyplot as plt

# --------- quality-of-life ----------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# --------- device ----------
device = torch.device("mps" if torch.backends.mps.is_available() else
                      ("cuda" if torch.cuda.is_available() else "cpu"))

# --------- model/tokenizer ----------
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

# ================================================
# Easy dataset
# ================================================
easy_name = "xTRam1/safe-guard-prompt-injection"
easy_ds = load_dataset(easy_name)  # has columns ["text","label"]

def preprocess_easy(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    return enc

easy_tok = easy_ds.map(preprocess_easy, batched=True)
easy_tok = easy_tok.rename_column("label", "labels")
easy_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ================================================
# Hard dataset
# ================================================
hard_name = "reshabhs/SPML_Chatbot_Prompt_Injection"
hard_ds = load_dataset(hard_name)  # columns: ["System Prompt","User Prompt","Prompt injection","Degree","Source"]

# make a train/test if none provided
if "test" not in hard_ds:
    hard_ds = hard_ds["train"].train_test_split(test_size=0.2, seed=SEED)

def preprocess_hard(batch):
    # join system+user into one text (handle None)
    sys = batch.get("System Prompt", [])
    usr = batch.get("User Prompt", [])
    text = [f"{(s or '')} {(u or '')}".strip() for s, u in zip(sys, usr)]
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=128)
    return enc

hard_tok = hard_ds.map(preprocess_hard, batched=True)
# labels column is "Prompt injection" (0/1); rename to "labels"
hard_tok = hard_tok.rename_column("Prompt injection", "labels")
# ensure labels are ints
for split in hard_tok:
    hard_tok[split] = hard_tok[split].cast_column("labels", hard_tok[split].features["labels"])
hard_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ================================================
# Merge easy + hard for training
# ================================================
combined_train = concatenate_datasets([easy_tok["train"], hard_tok["train"]])
# for evaluation during training we’ll use a combined dev set
combined_eval  = concatenate_datasets([easy_tok["test"],  hard_tok["test"]])

# ================================================
# Metrics
# ================================================
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# ================================================
# Train
# ================================================
training_args = TrainingArguments(
    output_dir="./results-combined",
    eval_strategy="epoch",          # NOTE: 4.57 uses eval_strategy (NOT evaluation_strategy)
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="./logs",
    logging_steps=100,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_train,
    eval_dataset=combined_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ================================================
# Evaluate separately on easy/hard test sets
# ================================================
easy_eval = trainer.evaluate(eval_dataset=easy_tok["test"])
hard_eval = trainer.evaluate(eval_dataset=hard_tok["test"])

print("\n=== Easy Test ===")
print(easy_eval)
print("\n=== Hard Test ===")
print(hard_eval)

# ================================================
# Confusion matrix + misclassified (hard set)
# ================================================
preds_out = trainer.predict(hard_tok["test"])
preds = np.argmax(preds_out.predictions, axis=1)
labels = preds_out.label_ids

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix — Hard Dataset")
plt.tight_layout()
plt.savefig("confusion_matrix_hard_dataset.png")
plt.close()

print("\nDetailed Classification Report (Hard):")
print(classification_report(labels, preds, digits=4))

# -----------------------------------------------
# Safely show misclassified examples
# -----------------------------------------------
raw_hard_test = hard_ds["test"]  # original (non-tokenized) split
cols = raw_hard_test.column_names

# get columns ONLY if they exist
user_prompts = raw_hard_test["User Prompt"] if "User Prompt" in cols else None
sys_prompts  = raw_hard_test["System Prompt"] if "System Prompt" in cols else None

def row_text(i):
    # Case 1: if User Prompt exists and is non-empty
    if user_prompts is not None and user_prompts[i] not in [None, ""]:
        return user_prompts[i]

    # Case 2: fallback → concatenate System + User prompt
    s = sys_prompts[i] if sys_prompts is not None else ""
    u = user_prompts[i] if user_prompts is not None else ""
    combined = (s + " " + u).strip()

    return combined if combined else "<no text available>"

wrong_idx = np.where(preds != labels)[0]
print(f"\nShowing {min(5, len(wrong_idx))} misclassified examples:\n")
for i in wrong_idx[:5]:
    txt = row_text(i)
    print(f"Text: {txt[:200]}...")
    print(f"True Label: {labels[i]} | Predicted: {preds[i]}\n")

# ================================================
# Final Summary
# ================================================
def p(v): return f"{v:.4f}"

print("\n========================================")
print(" PROMPT INJECTION CLASSIFICATION REPORT")
print("========================================")
print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
print("📘 Model Used:")
print(f"  {MODEL_NAME}\n")
print("Datasets:")
print(f"  1) {easy_name} (easy)")
print(f"  2) {hard_name} (hard)\n")
print("Results:")
print(f"  Easy  -> acc={p(easy_eval['eval_accuracy'])}  f1={p(easy_eval['eval_f1'])}")
print(f"  Hard  -> acc={p(hard_eval['eval_accuracy'])}  f1={p(hard_eval['eval_f1'])}")
drop = easy_eval["eval_accuracy"] - hard_eval["eval_accuracy"]
print(f"\nPerformance drop (hard vs easy): {p(drop)}")
print("Interpretation:")
print("  → If drop > 0.05, the model struggles more with adversarial/complex injections.")
print("========================================")
