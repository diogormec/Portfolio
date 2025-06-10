
# 🧬 Breast Cancer Diagnosis using Machine Learning

This project uses supervised machine learning techniques to predict whether a tumor is **malignant** or **benign** based on the features computed from digitized images of breast mass biopsies.

The dataset used is the widely-known **Breast Cancer Wisconsin Diagnostic Dataset**. The primary objective of this project is to explore different classification algorithms, evaluate their performance, and build a robust predictive model.

---

## 📁 Project Structure

```bash
breast-cancer-diagnosis-ml/
│
├── BrestCancer_BinaryClassif.ipynb  # Jupyter notebook with full workflow
├── wdbc.data.csv                    # Dataset (Breast Cancer Wisconsin)
└── README.md                        # Project documentation
```

---

## 📊 Features

- 🧹 Data Cleaning & Preprocessing  
- 📈 Exploratory Data Analysis (EDA)  
- 🔎 Feature Selection  
- 🤖 Multiple ML Models (Logistic Regression, Random Forest, SVM, KNN, XGBoost, etc.)  
- 📉 Dimensionality Reduction (PCA)  
- 📋 Evaluation Metrics (Confusion Matrix, ROC AUC, F1-score)

---

## 🧪 Models Trained

- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors  
- Support Vector Machine (SVM)  
- Stochastic Gradient Descent  
- Gradient Boosting  
- AdaBoost  
- XGBoost  
- Voting Classifier (Ensemble)

---

## 🧠 Key Findings

- The dataset is imbalanced, with more benign cases than malignant.
- Feature scaling and selection significantly improved model performance.
- Ensemble methods like **VotingClassifier** and **XGBoost** achieved excellent classification metrics.
- Dimensionality reduction (PCA) was used for better visualization and efficiency.

---

## 📉 Evaluation Metrics

- Accuracy  
- F1 Score  
- ROC AUC  
- Confusion Matrix  
- Classification Report

---

## 📚 Dataset

- Source: UCI Machine Learning Repository  
- Name: Breast Cancer Wisconsin (Diagnostic)  
- Link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

---

## 🚀 How to Run

1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/breast-cancer-diagnosis-ml.git
   cd breast-cancer-diagnosis-ml
   ```

2. Install required libraries  
   *(preferably in a virtual environment)*  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook  
   ```bash
   jupyter notebook BrestCancer_BinaryClassif.ipynb
   ```

---

## 📌 Requirements

- Python 3.7+
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost
- missingno

You can generate a `requirements.txt` file using:

```bash
pip freeze > requirements.txt
```

---

## 🤝 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- Scikit-learn Team  
- Kaggle & Open Source Contributors