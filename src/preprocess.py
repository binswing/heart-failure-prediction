import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer

def get_preprocessor(iscale=True):
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
    norm_numeric_features = ['Oldpeak']
    binary_features = ['FastingBS']
    nominal_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina']
    ordinal_features = ['ST_Slope']

    numeric_transformer = StandardScaler() if iscale else 'passthrough'
    norm_numeric_transformer = MinMaxScaler() if iscale else 'passthrough'
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('norm_num', norm_numeric_transformer, norm_numeric_features),
            ('bin', 'passthrough', binary_features),
            ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_features),
            ('ord', OrdinalEncoder(categories=[['Down', 'Flat', 'Up']]), ordinal_features),
        ],
        remainder='drop'
    )

def preprocess_data(df, iscale=True):
    df_clean = df.copy()
    df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, np.nan)
    df_clean['RestingBP'] = df_clean['RestingBP'].replace(0, np.nan)
    
    X = df_clean.drop('HeartDisease', axis=1)
    y = df_clean['HeartDisease']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)