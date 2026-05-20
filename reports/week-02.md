# Week 2 Progress Report

**What was completed this week:**
* Implemented text preprocessing (lowercasing, removing punctuation and special characters).
* Split the dataset into 80% training and 20% validation sets to prevent data leakage.
* Converted text sequences into numerical representations using `TfidfVectorizer` (max features = 10,000).
* Built a strong baseline model: a Logistic Regression wrapped in a `OneVsRestClassifier` to handle the multi-label nature of the dataset.
* Evaluated the baseline model using Macro ROC-AUC and a detailed classification report.

**Results & Insights so far:**
* **Baseline Macro ROC-AUC Score:** 0.9765
* The baseline performs very well on general toxicity (`toxic`, `obscene`). However, as seen in the classification report, the `recall` for rare classes like `threat` and `identity_hate` is very low. The model struggles to identify these classes due to the extreme class imbalance discovered in Week 1.

**Important commits or files changed:**
* `notebooks/02_baseline.ipynb`: Added the complete pipeline for data preparation, model training, and evaluation.
* `reports/week-02.md`: Added progress report.

**Problems or blockers:**
* The multi-label setup required a specific strategy (`OneVsRestClassifier`) since standard Logistic Regression only supports single-label multi-class problems. 
* Dealing with class imbalance is the main issue currently degrading model performance on rare tags.

**Plan for next week:**
* Start working with Deep Learning frameworks (PyTorch).
* Create a custom Dataset and DataLoader.
* Set up tokenization and word embeddings for Neural Networks.
* Train a deep learning model (BiLSTM or GRU) with `BCEWithLogitsLoss`.
