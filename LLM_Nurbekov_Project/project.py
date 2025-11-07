# ================================================
# Mini Real Project: Prompt Injection Classifier
# ================================================

# Model: protectai/deberta-v3-base-prompt-injection
# Dataset 1 (easy): xTRam1/safe-guard-prompt-injection
# Dataset 2 (hard): reshabhs/SPML_Chatbot_Prompt_Injection

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------------------------
# 1️⃣ Load pretrained model + tokenizer
# -----------------------------------------------
model_name = "protectai/deberta-v3-base-prompt-injection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# -----------------------------------------------
# 2️⃣ Load the first dataset (easy)
# -----------------------------------------------
dataset_name = "xTRam1/safe-guard-prompt-injection"
dataset = load_dataset(dataset_name)

# check column names to confirm what the text/label columns are called
print(dataset)

# -----------------------------------------------
# 3️⃣ Tokenize data
# -----------------------------------------------
def preprocess(example):
    return tokenizer(
        example["text"],  # change key if your column has a different name
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized = dataset.map(preprocess, batched=True)
tokenized = tokenized.rename_column("label", "labels")  # Hugging Face Trainer expects "labels"
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_data = tokenized["train"]
test_data = tokenized["test"]

# -----------------------------------------------
# 4️⃣ Define metrics
# -----------------------------------------------
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# -----------------------------------------------
# 5️⃣ Fine-tune on the easy dataset
# -----------------------------------------------
training_args = TrainingArguments(
    output_dir="./results-easy",
    eval_strategy="epoch",     # ✅ use this instead of evaluation_strategy
    save_strategy="epoch",
    per_device_train_batch_size=4,   # better for Apple Silicon memory
    per_device_eval_batch_size=4,
    num_train_epochs=1,              # shorter test run
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
easy_eval = trainer.evaluate()
print("\n Easy Dataset Results:", easy_eval)

# -----------------------------------------------
# 6️⃣ Evaluate the trained model on the harder dataset
# -----------------------------------------------
hard_dataset_name = "reshabhs/SPML_Chatbot_Prompt_Injection"
hard_dataset = load_dataset(hard_dataset_name)

# Split manually if there's no "test" split
if "test" not in hard_dataset:
    hard_dataset = hard_dataset["train"].train_test_split(test_size=0.2, seed=42)


def preprocess_hard(example):
    # Join system and user prompts into a single string
    text = [
        (sp or "") + " " + (up or "")
        for sp, up in zip(example["System Prompt"], example["User Prompt"])
    ]
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128
    )



hard_tokenized = hard_dataset.map(preprocess_hard, batched=True)
hard_tokenized = hard_tokenized.rename_column("Prompt injection", "labels")
hard_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

hard_eval = trainer.evaluate(eval_dataset=hard_tokenized["test"])
print("\n Hard Dataset Results:", hard_eval)
# ================================================
# 📊 Visualization: Confusion Matrix + Errors
# ================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch

# -----------------------------------------------
# 1️⃣ Get predictions on the hard dataset
# -----------------------------------------------
predictions = trainer.predict(hard_tokenized["test"])
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# -----------------------------------------------
# 2️⃣ Confusion matrix
# -----------------------------------------------
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion Matrix -> Hard Dataset")
plt.savefig("confusion_matrix_hard_dataset.png")
plt.close()

# -----------------------------------------------
# 3️⃣ Classification report (precision/recall/F1)
# -----------------------------------------------
print("Detailed Classification Report:")
print(classification_report(labels, preds, digits=4))

# -----------------------------------------------
# 4️⃣ Show misclassified examples
# -----------------------------------------------
texts = hard_dataset["test"]["text"]
wrong_idx = np.where(preds != labels)[0]

print(f"\n Showing {min(5, len(wrong_idx))} misclassified examples:\n")
for i in wrong_idx[:5]:
    print(f"Text: {texts[i][:200]}...")  # truncate long text
    print(f"True Label: {labels[i]} | Predicted: {preds[i]}\n")

# ================================================
# 🧾 Final Summary Report
# ================================================
from datetime import datetime

def print_summary(easy_eval, hard_eval):
    print("========================================")
    print(" PROMPT INJECTION CLASSIFICATION REPORT")
    print("========================================")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("📘 Model Used:")
    print("  protectai/deberta-v3-base-prompt-injection\n")

    print("Datasets Evaluated:")
    print("1  xTRam1/safe-guard-prompt-injection (easy)")
    print("2 reshabhs/SPML_Chatbot_Prompt_Injection (hard)\n")

    print(" Easy Dataset Results:")
    print(f"  Accuracy: {easy_eval['eval_accuracy']:.4f}")
    print(f"  F1 Score: {easy_eval['eval_f1']:.4f}\n")

    print(" Hard Dataset Results:")
    print(f"  Accuracy: {hard_eval['eval_accuracy']:.4f}")
    print(f"  F1 Score: {hard_eval['eval_f1']:.4f}\n")

    diff_acc = easy_eval['eval_accuracy'] - hard_eval['eval_accuracy']
    print(f" Performance Drop (Hard vs Easy): {diff_acc:.4f}")
    print("\n Interpretation:")
    if diff_acc > 0.05:
        print("  → The model struggles more with complex or adversarial prompt injections.")
    else:
        print("  → The model generalizes well across both datasets.")
    print("========================================")

print_summary(easy_eval, hard_eval)
