GitHub: https://github.com/2026ak05729-cyber/Bank-Marketing-Classification
Streamlit: https://2026ak05729-cyber-bank-marketing-classification-app-rdnfyg.streamlit.app/#comparison-of-all-five-models
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
MODEL NAME	ACCURACY	AUC	PRECISION	RECALL	F1	MCC
LOGISTIC REGRESSION	0.8301	0.8003	0.3589	0.6455	0.4613	0.3926
DECISION TREE	0.8468	0.6275	0.3282	0.3438	0.3358	0.2493
KNN	0.8913	0.7276	0.5366	0.2608	0.3510	0.3227
NAIVE BAYES	0.8570	0.7762	0.3890	0.4720	0.4265	0.3478
SVM	0.8311	0.8001	0.3610	0.6476	0.4636	0.3953

Observations
MODEL NAME	OBSERVATIONS
LOGISTIC REGRESSION	Logistic Regression achieved an Accuracy of 83.01% and an AUC of 0.8003. After applying balanced class weights, Recall improved significantly to 64.55%, compared with 20.80% in the baseline model. The F1 score also improved from 0.3151 to 0.4613. 
DECISION TREE	The Decision Tree achieved an Accuracy of 84.68% and an AUC of 0.6275. Its Recall was 34.38%, with an F1 score of 0.3358. Applying balanced class weights produced only a marginal improvement in F1 and MCC. Low Accuracy made it unusable.
KNN	KNN achieved an Accuracy of 89.13%, with an AUC of 0.7276. However, its Recall was only 26.08%, indicating that it missed a substantial proportion of actual subscribers and not very usable
NAIVE BAYES	Naive Bayes achieved an Accuracy of 85.70% and an AUC of 0.7762. It recorded a Recall of 47.20% and an F1 score of 0.4265. Still this model is not good enough for identifying subscribers due to low recall of 47.20%. 
SVM	SVM achieved an Accuracy of 83.11% and an AUC of 0.8001. After applying balanced class weights, Recall increased substantially from 20.15% to 64.76%, while the F1 score improved from 0.3109 to 0.4636. SVM also achieved the highest MCC of 0.3953 among the five classifiers. This indicates a better overall classification balance despite the reduction in Accuracy, the relatively higher Recall makes it the best model of all.
OVERALL WINNER	Since the dataset is significantly imbalanced, Accuracy alone is not an appropriate criterion for selecting the best classifier. Recall, F1 and MCC provide more meaningful insights into the models' ability to identify potential term deposit subscribers. Therefore, SVM is selected as the overall winner, considering the objective of identifying as many potential subscribers as possible.


## Running the Application
Install the dependencies using:

`pip install -r requirements.txt`

Run the application using:

`streamlit run app.py`

The application allows users to upload a labelled test CSV, select a classification model and view its evaluation metrics and confusion matrix.
