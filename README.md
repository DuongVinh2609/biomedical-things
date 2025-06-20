# Biomedical Machine Learning Projects

This repository contains introductory machine learning projects focused on **biomedical data analysis**, implemented using Python and scikit-learn. These projects were created as part of my initial learning in ML and its applications in biomedical fields.

## 📁 Projects Included

### 1. 🔬 Breast Cancer Classification

> Predicting whether a breast tumor is **malignant (M)** or **benign (B)** using classical machine learning classifiers.

- **Input**: 30 numeric features  
- **Label Encoding**:  
  - `0`: Malignant  
  - `1`: Benign

- **Algorithms used**:
  - SVM (Support Vector Machine)
  - Perceptron
  - Passive Aggressive Classifier
  - Multi-layer Perceptron (Neural Network)
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Decision Tree

- **Evaluation**:  
  Each model is evaluated using:
  - Precision
  - Recall
  - F1-score

- **Preprocessing**:
  - `StandardScaler` and `MinMaxScaler` via scikit-learn `Pipeline`

---

### 2. 🍬 Diabetes Prediction

> Predicting whether a patient is likely to have diabetes based on health diagnostic measurements.

- **Dataset**: Pima Indians Diabetes Dataset
- **Input**: 8 numeric clinical features (e.g., BMI, glucose level)
- **Label**:  
  - `1`: Diabetic  
  - `0`: Non-diabetic

- **Model used**:
  - Multi-layer Perceptron (MLPClassifier)
    - One hidden layer (25 neurons)
    - Activation: Identity
    - Optimizer: SGD
    - Max iterations: 5000

- **Preprocessing**:
  - Data normalized using `StandardScaler`

- **Metrics**:
  - Precision
  - Recall
  - F1-score

---

## 🛠 Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (ML models, scalers, evaluation)

---


## 📌 Notes & Future Work

- Apply cross-validation for better model robustness.
- Tune hyperparameters via grid/random search.
- Expand to deep learning (e.g., TensorFlow/Keras) for more complex biomedical tasks.
- Add visualization of metrics (confusion matrix, ROC curves).

---

## 📖 Author

These projects were created during self-study in **Machine Learning** and **Biomedical Informatics**.

