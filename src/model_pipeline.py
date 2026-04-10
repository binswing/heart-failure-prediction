import joblib
import pandas as pd
from sklearn.impute import KNNImputer
from src.preprocess import get_preprocessor

class HeartDiseasePipeline:
    def __init__(self, model_obj):
        self.model = model_obj
        self.preprocessor = get_preprocessor(iscale=True)
        self.imputer = KNNImputer(n_neighbors=5, weights='distance')
        self.feature_names = None

    def fit(self, X_train, y_train):
        X_proc = self.preprocessor.fit_transform(X_train)
        self.feature_names = self.preprocessor.get_feature_names_out()

        X_df = pd.DataFrame(X_proc, columns=self.feature_names)
        X_final = self.imputer.fit_transform(X_df)

        self.model.fit(X_final, y_train)

    def predict(self, X):
        X_proc = self.preprocessor.transform(X)
        X_df = pd.DataFrame(X_proc, columns=self.feature_names)
        X_final = self.imputer.transform(X_df)
        return self.model.predict(X_final)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)