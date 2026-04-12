import pandas as pd
import os
from src.preprocess import preprocess_data
from src.model_pipeline import HeartDiseasePipeline
from src.utils import load_config, full_evaluation
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/raw/heart.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)
config = load_config()

models = {
    "LogReg": LogisticRegression(**config['logistic_regression']),
    "XGB": XGBClassifier(**config['xgboost']),
    "SVM": SVC(**config['svm'])
}

for name, m_obj in models.items():
    pipeline = HeartDiseasePipeline(m_obj)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    full_evaluation(name, y_test, preds)
    pipeline.save(f"models/{name}_model.joblib")