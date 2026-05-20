# Week 3 Progress Report

**What was completed this week:**
* Transitioned from traditional Machine Learning to Deep Learning using PyTorch.
* Developed a custom tokenization pipeline, building a vocabulary of the top 15,000 words.
* Implemented a PyTorch `Dataset` and `DataLoader` to handle batching and sequence padding (fixed length of 100 tokens).
* Designed a Bidirectional LSTM (BiLSTM) model with an Embedding layer and a fully connected output layer.
* Wrote a custom training loop using `BCEWithLogitsLoss` (optimized for multi-label tasks) and the Adam optimizer.
* Trained the network and evaluated its performance on the validation set.

**Results & Insights so far:**
* **BiLSTM Macro ROC-AUC Score:** 0.9645
* The deep learning model successfully learns representations of toxic text. The training loop effectively minimizes the binary cross-entropy loss across epochs. 

**Important commits or files changed:**
* `notebooks/03_deep_learning.ipynb`: Added the complete deep learning pipeline (DataLoaders, Architecture, Training Loop, Evaluation).
* `reports/week-03.md`: Added progress report.

**Problems or blockers:**
* PyTorch requires manual tensor shape management (especially concatenating bidirectional hidden states), which was a bit challenging.
* Training an RNN on CPUs is very slow. I used a subset of the dataset (50%) to speed up the local experimental phase. 

**Plan for next week (Final Week):**
* Finalize the evaluation (compare Baseline vs. Deep Learning).
* Perform Error Analysis (find a success case and a failure case).
* Write the comprehensive Final Report (`final-report.md`).
* Prepare the presentation slides/demo for the project defense.
