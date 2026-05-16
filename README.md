# Deep Learning for Multi-Label Toxic Comment Classification 

## 📌 Project Overview
This project applies deep learning techniques to identify and classify toxic speech in online comments. The goal is to build a multi-label classification model that can predict six different toxicity categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

## 📂 Dataset
The project uses the **Jigsaw Toxic Comment Classification Challenge** dataset.
* **Source:** [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) / [Hugging Face](https://huggingface.co/datasets/martin-ha/toxic-comment-model)
* **Size:** ~159k training examples, ~153k test examples.
* **Format:** Text strings mapping to 6 binary labels.

> **Note:** The raw data files are not hosted in this repository. Please see `data/README.md` for download instructions.

## 🛠️ Planned Methodology
1. **Baseline Model:** TF-IDF Vectorization + Logistic Regression (One-vs-Rest).
2. **Deep Learning Model:** Bidirectional LSTM (BiLSTM) or GRU with pre-trained embeddings / Transformer fine-tuning.
3. **Evaluation:** ROC-AUC and Macro F1-score to handle class imbalance.

## 📅 Weekly Progress
* [Week 1 Report](reports/week-01.md): Repository setup, dataset selection, and Exploratory Data Analysis (EDA).
* *Week 2 Report: (Coming soon)*
* *Week 3 Report: (Coming soon)*
* *Week 4 Report: (Coming soon)*

## 🚀 How to Run
*(Instructions will be added as code is developed)*
