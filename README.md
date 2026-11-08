# Bank Marketing Classification

## Project Overview
The data is related to direct marketing campaigns by a bank. The marketing campaigns were based on demographic, financial and other marketing campaign information from phone calls.
The classification goal is to predict if the client will subscribe (yes/no) a term deposit variable y).
We will build a logistic regression along side Decision Tree, KNN, Naive Bayes and SVM to predict y.
For evaluation of the models, we will use the metrics Accuracy, AUC, Precision, Recall, F1 and MCC. Out of these metrics, for this exercise Recall is the most important, details in the next section.

## Dataset and Preprocessing
Dataset Name: “Bank Marketing”
Dataset Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
Samples: 41176 after removing 12 duplicated rows. No NaN values.
Features: 20 features, 10 numerical and 10 categorical. Post encoding, 62 features.
Target: a binary classification with 2 outcomes – yes for subscribed and no for not subscribed
Class distribution: 88.7% belong to class ‘no’ and 11.3% belong to ‘yes’ class.
Data splitting: Training and test set split at 80:20 ratio.
Metric Justification: Recall is the most important because in this dataset, the chances of a client subscribing is only 11%, so a false negative means failing to identify a client who would actually subscribe. Prioritizing recall means we aim to minimize these missed opportunities.

## Machine Learning Models
Five classification algorithms were implemented:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Class weighting was used for Logistic Regression, Decision Tree and SVM to address class imbalance.

## Evaluation Metrics
Models were evaluated using:
- Accuracy
- ROC-AUC
- Precision
- Recall (This measure is the Priority)
- F1-score
- Matthews Correlation Coefficient (MCC)

## Repository Structure
- `app.py` – Streamlit application
- `models/` – Saved trained classification models
- `preprocessing/` – Saved preprocessing artifacts
- `test_data.csv` – Test dataset used for model evaluation
- `requirements.txt` – Python dependencies

## Results
SVM achieved the best Recall (0.6476), F1-score (0.4636) and MCC (0.3953), marginally outperforming Logistic Regression on these metrics.

## Running the Application
Install the dependencies using:

`pip install -r requirements.txt`

Run the application using:

`streamlit run app.py`

The application allows users to upload a labelled test CSV, select a classification model and view its evaluation metrics and confusion matrix.
