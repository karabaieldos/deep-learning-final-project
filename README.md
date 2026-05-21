# Deep Learning for Multi-Label Toxic Comment Classification 

##  Project Overview
This project applies deep learning techniques to identify and classify toxic speech in online comments. The goal is to build a multi-label classification model that can predict six different toxicity categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

##  Dataset
The project uses the **Jigsaw Toxic Comment Classification Challenge** dataset.
* **Source:** [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) / [Hugging Face](https://huggingface.co/datasets/dirtycomputer/Toxic_Comment_Classification_Challenge)
* **Size:** ~159k training examples, ~153k test examples.
* **Format:** Text strings mapping to 6 binary labels.

> **Note:** The raw data files are not hosted in this repository. Please see `data/README.md` for download instructions.

##  Planned Methodology
1. **Baseline Model:** TF-IDF Vectorization + Logistic Regression (One-vs-Rest).
2. **Deep Learning Model:** Bidirectional LSTM (BiLSTM) or GRU with pre-trained embeddings / Transformer fine-tuning.
3. **Evaluation:** ROC-AUC and Macro F1-score to handle class imbalance.

##  Weekly Progress
* [Week 1 Report](reports/week-01.md): Repository setup, dataset selection, and EDA.
* [Week 2 Report](reports/week-02.md): Preprocessing and Baseline Model.
* [Week 3 Report](reports/week-03.md): Deep Learning BiLSTM Model.
* [Week 4 Report](reports/week-04.md): Error Analysis and Final Report.

##  Final Submission
Please refer to the [**Final Project Report**](final-report.md) for complete details on architecture, results, and error analysis.
