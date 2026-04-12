# Heart Failure Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-green)

## Introduction
This project is part of the **Khai phá Dữ liệu (CO3029)** course at **Ho Chi Minh City University of Technology (HCMUT)**.

The goal of this project is to apply Data Mining and Machine Learning techniques to predict the risk of **Heart Disease** in patients. By analyzing clinical and demographic data (such as Age, Max Heart Rate, Chest Pain Type, ST Slope, etc.), we aim to quantify the impact of these factors and build a reliable diagnostic support tool.

The project features a complete end-to-end pipeline from data cleaning and Exploratory Data Analysis (EDA) to model training (Logistic Regression, XGBoost, SVM) and deployment via an interactive  **Streamlit Web Application** featuring a multi-model consensus mechanism.

**Kaggle notebook:** [Click here to view the Kaggle notebook](https://www.kaggle.com/code/longnguyentuan/heart-failure-prediction-dm)

**Github Repo:** [Click here to view the Github Repo](https://github.com/binswing/Pisa-scores-prediction)

**Live Demo:** [Click here to view the App](https://heart-failure-prediction-demo.streamlit.app/)

---

## Team Members (L03 - Team 05)
**Instructor:** Mr. Đỗ Thanh Thái

* **Nguyễn Trần Nhật Châu** - 2310335
* **Nguyễn Tuấn Long** - 2311915
* **Huỳnh Huy Hoàng** - 2311041

---

## Dataset
* **Source:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)
* **Scope:** 918 Patient records with 11 clinical feature.
* **Key Features:**
    * `Age & Sex`: Patient demographics.
    * `ChestPainType`: Type of chest pain (ATA, NAP, ASY, TA).
    * `RestingBP & Cholesterol`: Vital signs and blood metrics.
    * `MaxHR`: Maximum heart rate achieved during exercise.
    * `ST_Slope`: The slope of the peak exercise ST segment (Up, Flat, Down) - a highly critical risk predictor.
    * **Target:** `HeartDisease` (1: Presence of heart disease, 0: Normal/Healthy).

---

## Tech Stack & Methodology

### 1. Data Preprocessing
* **Cleaning:** Handled biologically impossible zero values (e.g., 0 Cholesterol or Resting BP) by converting them to NaN to prevent data distortion.
* **Imputation:** Used KNN Imputer (k=5) to handle missing values based on patient similarity rather than blunt averages.
* **Scaling & Encoding:** Centralized in a **ColumnTransformer** using **StandardScaler** for continuous features, MinMaxScaler for normalized features, and **OneHotEncoder/OrdinalEncoder** for categorical symptoms to prevent data leakage.
* **Splitting:** Stratified Train/Test split (80:20) based on the target variable to ensure distribution consistency.

### 2. Machine Learning Models
We implemented and compared three classification algorithm:
* **Logistic Regression:** A highly interpretable linear baseline model.
* **XGBoost:** A robust framework using level-wise growth, regularization ($L_1, L_2$) to prevent overfitting, and parallel processing.
* **Support Vector Machine (SVM):** Configured with a linear kernel to find the optimal diagnostic hyperplane.

### 3. Application
* **Framework:** Streamlit
* **Features:**
    * **Interactive Clinical Dashboard**: Data visualization including demographic Violin plots, vital sign Boxplots, and Multivariate Correlation Heatmaps.
    * **Model Consensus Inference**: Real-time diagnostic interface that runs patient data through all three AI engines simultaneously to reach a "consensus" prediction.
    * **Test Set Auditing**: A tool to scroll through unseen patient records and cross-reference Ground Truth against AI predictions (Correct/Incorrect highlighting).

## Installation
- Build
```bash
    #Create venv
    python -m venv .venv
    
    #Activate venv
    source .venv/bin/activate #Linux/Mac
    .venv\Scripts\activate #Windows
    
    #Install requirements
    pip install -r requirements.txt
```
- Run demo app locally
```bash
    python main.py #Run this line once to generate the .joblib model files 
    streamlit run visualization_and_demo_app/app.py
```