# Prompt Injection Classifier — Final Project

This project fine‑tunes a pretrained model to detect **prompt injection attacks**.  
We evaluate performance on:

1. **Easy Dataset:** `xTRam1/safe-guard-prompt-injection`
2. **Hard Dataset:** `reshabhs/SPML_Chatbot_Prompt_Injection`

---

## 🔍 Project Goal

Each team:
- Downloads a Hugging Face model  
- Chooses **two datasets** (easy + hard)  
- Fine‑tunes the model  
- Evaluates accuracy + F1  
- Reports scores and misclassified examples  

Your model learned from **both datasets combined** and achieved:

| Dataset | Accuracy | F1 Score |
|--------|----------|----------|
| Easy   | **98.23%** | **98.19%** |
| Hard   | **74.33%** | **68.81%** |

---

## 📦 Environment Setup

```bash
conda create -n promptenv python=3.10 -y
conda activate promptenv

pip install torch torchvision torchaudio
pip install transformers datasets scikit-learn matplotlib seaborn
pip install accelerate
```

If you are using a Mac with Apple Silicon:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## ▶️ How to Run

Inside the project folder:

```bash
python project.py
```

Outputs generated:

- `confusion_matrix_hard_dataset.png`
- Terminal logs with accuracy & F1
- Misclassified examples printed at the end
- A summary report

---

## 📊 What the Script Does

### 1. Loads pretrained model  
`protectai/deberta-v3-base-prompt-injection`

### 2. Loads both datasets  
- Easy → already in `text`,`label` format  
- Hard → merges `"System Prompt"` + `"User Prompt"` into one field  

### 3. Combines datasets  
Model sees both easy and hard samples during training.

### 4. Fine-tunes for 3 epochs  
Improves generalization.

### 5. Evaluates  
Prints:
- accuracy  
- F1 score  
- classification report  
- confusion matrix image  
- misclassified prompts  

---

## 📁 File Structure

```
LLM_Nurbekov_Project/
│── project.py
│── run_prompt_project.sh
│── logs/
│── results-combined/
│── confusion_matrix_hard_dataset.png
│── prompt_logs.txt
│── prompt_errors.txt
└── README.md   ← (this file)
```

---

## 🧠 Interpretation

- Easy dataset accuracy is very high → the model understands straightforward injection attacks.
- Hard dataset accuracy is lower → adversarial, tricky prompts make classification harder.
- Combined training significantly improved hard dataset performance compared to training only on the easy one.

---

## ✨ Summary

This project successfully:
- Fine‑tuned a DeBERTa model  
- Handled two different prompt‑injection datasets  
- Improved robustness by merging datasets  
- Produced evaluation metrics + confusion matrix  
- Printed top misclassified examples  

The model performs **very well** overall and demonstrates strong generalization.

---

## 🚀 Future Improvements

- Add more adversarial samples  
- Try larger models (DeBERTa‑Large, Llama‑Guard, Roberta‑Large)  
- Use contrastive loss  
- Add prompt‑context augmentation  
- Train for more epochs on cluster GPUs  

---

