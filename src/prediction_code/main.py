import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, mean_absolute_error, roc_auc_score, RocCurveDisplay
from sklearn.compose import ColumnTransformer
plt.style.use('seaborn-v0_8-whitegrid')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")

path = "/dataset/heart.csv"
df = pd.read_csv(os.getcwd()+path)

print("\n===== SHAPE, COLUMNS =====")
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\n===== HEAD =====")
print(df.head())

print("\n===== INFO =====")
print(df.info())

def preprocess_final(df, test_size=0.2, random_state=42, iscale=True):
    # 1. Cleaning
    df_clean = df.copy()
    df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, np.nan)
    df_clean['RestingBP'] = df_clean['RestingBP'].replace(0, np.nan)
    
    # 2. Split
    X = df_clean.drop('HeartDisease', axis=1)
    y = df_clean['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. Define Transformers
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
    norm_numeric_features = ['Oldpeak']
    binary_features = ['FastingBS']
    nominal_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina']
    ordinal_features = ['ST_Slope']

    numeric_transformer = StandardScaler() if iscale else 'passthrough'
    norm_numeric_transformer = MinMaxScaler() if iscale else 'passthrough'
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('norm_num', norm_numeric_transformer, norm_numeric_features),
            ('bin', 'passthrough', binary_features),
            ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_features),
            ('ord', OrdinalEncoder(categories=[['Down', 'Flat', 'Up']]), ordinal_features),
        ],
        remainder='drop'
    )

    # 4. Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 5. Get Feature Names Safely
    all_feature_names = preprocessor.get_feature_names_out()
    
    X_train_combined = pd.DataFrame(X_train_processed, columns=all_feature_names)
    X_test_combined = pd.DataFrame(X_test_processed, columns=all_feature_names)

    # 6. Impute
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_train_imputed = imputer.fit_transform(X_train_combined)
    X_test_imputed = imputer.transform(X_test_combined)
    
    X_train_final = pd.DataFrame(X_train_imputed, columns=all_feature_names)
    X_test_final = pd.DataFrame(X_test_imputed, columns=all_feature_names)
    
    return X_train_final, X_test_final, y_train, y_test
X_train, X_test, y_train, y_test = preprocess_final(df,iscale = True)

def model_train(model):
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    print("Accuracy: {0:.4f}".format(accuracy_score(y_test, y_pred)))
    print("Precision: {0:.4f}".format(precision_score(y_test, y_pred)))
    print("Recall: {0:.4f}".format(recall_score(y_test, y_pred)))

    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    print("Cross Validation Score : ",'{0:.4f}'.format(cross_val_score(model,X_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("ROC_AUC Score : ",'{0:.4f}'.format(roc_auc_score(y_test,y_pred)))
    # RocCurveDisplay.from_estimator(model, X_test,y_test)
    # plt.title('ROC_AUC_Plot')
    # plt.show()

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
print("--- KNN ---")
model_train(knn)

# Train RandomForest Classifier
rf = RandomForestClassifier(n_estimators=120, random_state=42)
print("--- RandomForest ---")
model_train(rf)

# Train MLP Classifier
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
print("--- MLP ---")
model_train(nn)

# Train Logistic Regression 
lr = LogisticRegression(random_state=42, max_iter=1000)
print("--- Logistic Regression ---")
model_train(lr)

# Train XGBoost
xgb_model = XGBClassifier(
                        random_state=42,
                        use_label_encoder=False, 
                        eval_metric='logloss',
                        n_estimators=120,
                        learning_rate=0.1,
                        max_depth=3,
                        max_leaves=4,
                        reg_alpha=0.5,
                        reg_lambda=1,
                        )
print("--- XGBoost ---")
model_train(xgb_model)

# Train SVM
svm_model = SVC(kernel='rbf', random_state=42)
print("--- SVM ---")
model_train(svm_model)