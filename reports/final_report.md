# Final Project Report: Multi-Label Toxic Comment Classification

## 1. Project Title
Deep Learning for Multi-Label Toxic Comment Classification.

## 2. Problem Statement
The goal of this project is to build an automated moderation model capable of detecting and classifying toxic speech in online comments across six categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. This is crucial for maintaining safe online communities and reducing the manual workload on human moderators.

## 3. Dataset Description
* **Source:** Jigsaw Toxic Comment Classification Challenge (Kaggle).
* **Size:** ~159,000 training comments.
* **Format:** Text strings with 6 binary target columns.
* **Insights:** The dataset suffers from extreme class imbalance. About 90% of the comments are completely clean, and some classes (like `threat`) have very few positive examples.

## 4. Data Preprocessing
* Converted all text to lowercase.
* Expanded basic English contractions (e.g., "it's" to "it is").
* Removed all special characters, numbers, and punctuation using Regular Expressions.
* Stripped extra whitespaces.
* Split data into 80% Train and 20% Validation sets to prevent data leakage.

## 5. Model Architectures & Training Setup
**Baseline Model:**
* TF-IDF Vectorizer (top 10,000 features).
* Logistic Regression wrapped in a `OneVsRestClassifier`.

**Deep Learning Model:**
* Custom Vocabulary (top 15,000 words) with fixed sequence length (100 tokens).
* **Architecture:** Embedding Layer (128 dim) -> Bidirectional LSTM (64 hidden dim) -> Fully Connected Linear Layer (6 outputs).
* **Optimizer & Loss:** Adam Optimizer (lr=0.001) and `BCEWithLogitsLoss` (optimized for multi-label tasks).

## 6. Evaluation Metrics & Results
Due to class imbalance, **Macro ROC-AUC** was chosen as the primary metric, as it treats all classes equally regardless of their support size.

| Model | Macro ROC-AUC Score |
| :--- | :--- |
| **Baseline (TF-IDF + LogReg)** | 0.9765 |
| **Deep Learning (BiLSTM)** | 0.9645 |

## 7. Error Analysis
* **Success Case:** The model easily identifies explicit toxicity and profanity. It correctly flagged multiple labels simultaneously when trigger words were present.
* **Failure Case:** The model struggles heavily with **sarcasm** and **implicit bias**. Comments using positive words in a mocking tone (e.g., "Oh, brilliant idea Einstein") often trick the model into predicting the text as "clean" (False Negative).

## 8. Limitations & Future Work
* **Limitations:** The fixed sequence length truncates very long comments, potentially losing important context. The BiLSTM does not capture deep bidirectional context as well as transformer models.
* **Next Steps:** Fine-tuning a pre-trained Transformer model like `DistilBERT` or `RoBERTa` would likely solve the sarcasm issue and significantly boost recall for rare classes like `threat`.
