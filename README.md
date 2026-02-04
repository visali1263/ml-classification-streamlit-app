# ml-classification-streamlit-app
1. Problem Statement

The objective of this assignment is to implement multiple supervised machine learning classification models on a single dataset, evaluate their performance using standard evaluation metrics, and deploy an interactive web application using Streamlit to demonstrate the trained models.


2. Dataset Description 

Dataset Name: Breast Cancer Wisconsin (Diagnostic)

Source: UCI Machine Learning Repository

Problem Type: Binary Classification

Number of Instances: 569

Number of Features: 30 numeric features

Target Variable: target

0 → Malignant

1 → Benign


3. Models Used & Evaluation Metrics

The following six classification models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

Each model was evaluated using the following metrics:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

<img width="784" height="272" alt="image" src="https://github.com/user-attachments/assets/79686f92-4d96-45e7-92f6-ea7cf20d1a14" />



4. Observations on Model Performance (3 Marks)
   
<img width="1300" height="240" alt="image" src="https://github.com/user-attachments/assets/0cdad0a1-1fcc-4d4e-aeca-eb8c81fad338" />


NOTE : The scaler used during training is saved and reused during inference to maintain feature consistency.


