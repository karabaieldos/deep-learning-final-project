# Deep Learning for Multi-Label Toxic Comment Classification

## 1. Project Title
**Deep Learning for Multi-Label Toxic Comment Classification**

---

## 2. Problem Statement
* **What problem are you trying to solve?** The goal of this project is to build an automated content moderation system capable of detecting and classifying toxic speech in online comments across six distinct categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
* **Why is this problem useful or interesting?** Online platforms heavily struggle with toxic behavior, which toxic content directly harms user engagement and communities. Automated moderation tools are essential to process millions of messages efficiently, reduce the manual workload on human moderators, and maintain safe digital spaces.
* **What will the model predict or generate?** Given a raw text string (a user comment), the model will predict independent probabilities for each of the six toxicity labels, treating it as a multi-label classification task.

---

## 3. Dataset
* **Dataset Name:** Jigsaw Toxic Comment Classification Challenge
* **Dataset Source:** [Kaggle Dataset Page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
* **Number of Examples:** ~159,571 training comments
* **Input Features:** Raw text strings (English comments from Wikipedia talk pages)
* **Target Labels:** 6 binary columns (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`)
* **Data Format:** `.csv` files
* **License / Usage Notes:** Released under CC0 (Public Domain) for competition and academic research.

---

## 4. Planned Method
* **Simple Baseline:** Text preprocessing + TF-IDF Vectorization (top 10,000 features) followed by a Logistic Regression model. To handle the multi-label nature of the task, the model is wrapped in a `OneVsRestClassifier`.
* **Deep Learning Model:** A Bidirectional LSTM (BiLSTM) network implemented in PyTorch, utilizing a custom embedding layer (128 dimensions), a hidden layer size of 64, and a fully connected linear output layer for the 6 target categories.
* **Loss Function:** `BCEWithLogitsLoss` (Binary Cross-Entropy with Logits), which combines a Sigmoid layer and standard BCELoss for numerically stable multi-label classification.
* **Evaluation Metrics:** **Macro ROC-AUC** serves as the primary metric due to severe class imbalance. A secondary classification report (Precision, Recall, F1-score) will be tracked per class.
* **Data Split Plan:** The training dataset will be split into **80% Train** and **20% Validation** sets to tune hyperparameters and monitor overfitting.

---

## ⚠️ 5. Expected Challenges
> *During the implementation phase, we anticipate and plan to address the following core difficulties:*

1. **Extreme Class Imbalance:** Approximately 90% of the comments are completely clean. Minority classes like `threat` or `identity_hate` have very few positive instances, making optimization difficult.
2. **Noisy Text Data:** Internet comments are filled with typos, internet slang, and intentional obfuscations designed to bypass standard filters, which complicates tokenization.
3. **Implicit Toxicity and Sarcasm:** Detecting sarcastic comments where individual words are positive but the overall semantic meaning is derogatory remains a substantial roadblock.
4. **Computational Resources:** Training recurrent neural networks like a BiLSTM on over 150,000 sequences is computationally intensive and requires GPU acceleration.

---

## 6. Weekly Plan

| Week | Planned Work | Expected Output |
| :--- | :--- | :--- |
| **Week 1** | Repository setup, dataset acquisition, and deep Exploratory Data Analysis (EDA). Analyze class distribution and sequence lengths. | Structured GitHub repository, comprehensive `README.md`, and initialized EDA notebook. |
| **Week 2** | Text cleaning/preprocessing pipeline implementation (regex-based cleaning, lowercase). Build and evaluate the statistical baseline. | Baseline notebook containing TF-IDF and Logistic Regression pipeline with benchmark metrics. |
| **Week 3** | Develop deep learning components using PyTorch. Build vocabulary, custom Dataset/DataLoader, and the BiLSTM network. Run training loops. | Deep learning notebook containing the fully trained BiLSTM architecture and validation metrics. |
| **Week 4** | Conduct rigorous error analysis on success/failure cases. Aggregate results, format final documentation, and prepare presentation. | Error analysis notebook, finalized project report (`final-report.md`), and clean repository. |
