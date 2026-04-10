import streamlit as st
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.preprocess import preprocess_data
from src.model_pipeline import HeartDiseasePipeline


def render_training_tab(df):
    st.title("Multi-Model Inference & Comparison")

    model_files = {
        "Logistic Regression": "models/LogReg_model.joblib",
        "XGBoost": "models/XGB_model.joblib",
        "SVM": "models/SVM_model.joblib"
    }

    if 'pipelines' not in st.session_state:
        st.session_state['pipelines'] = {}
        with st.spinner("Loading AI Engines and Preparing Data..."):
            for name, path in model_files.items():
                if os.path.exists(path):
                    st.session_state['pipelines'][name] = HeartDiseasePipeline.load(path)
                else:
                    st.error(f" Missing {name}! Run `train_and_save.py` first.")
                    return

            X_train, X_test, y_train, y_test = preprocess_data(df)
            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test

    pipelines = st.session_state['pipelines']
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']

    tab_metrics, tab_inference, tab_test = st.tabs(["Leaderboard", "Compare Single Inference", "Inspect Predictions"])

    # === TAB 1: METRICS LEADERBOARD ===
    with tab_metrics:
        st.subheader("Model Performance Leaderboard")
        st.markdown("Metrics calculated on the unseen test set, with a 5-Fold Cross Validation score on the training set.")
        
        if 'leaderboard_df' not in st.session_state:
            with st.spinner("Calculating Cross-Validation and ROC AUC scores..."):
                results = []
                for name, pipe in pipelines.items():
                    preds = pipe.predict(X_test)
                    
                    X_train_proc = pipe.preprocessor.transform(X_train)
                    X_train_df = pd.DataFrame(X_train_proc, columns=pipe.feature_names)
                    X_train_final = pipe.imputer.transform(X_train_df)

                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_auc = cross_val_score(pipe.model, X_train_final, y_train, cv=cv, scoring='roc_auc').mean()
                    
                    results.append({
                        "Model": name,
                        "CV ROC-AUC (Train)": cv_auc,
                        "ROC-AUC (Test)": roc_auc_score(y_test, preds),
                        "Accuracy": accuracy_score(y_test, preds),
                        "Precision": precision_score(y_test, preds),
                        "Recall": recall_score(y_test, preds),
                        "F1 Score": f1_score(y_test, preds)
                    })
                
                st.session_state['leaderboard_df'] = pd.DataFrame(results).set_index("Model")

        results_df = st.session_state['leaderboard_df']
        
        styled_df = (
            results_df.style
            .format("{:.4f}")
            .highlight_max(axis=0, props="background-color: #1f77b4; color: white;")
        )
        
        st.dataframe(styled_df, width='stretch')

    # === TAB 2: SINGLE INFERENCE COMPARISON ===
    with tab_inference:
        st.subheader("Predict Risk for a New Patient")
        
        with st.form("inference_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", 20, 100, 50)
                sex = st.selectbox("Sex", ["M", "F"])
                cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
                bp = st.number_input("Resting BP", 80, 200, 120)
            with col2:
                chol = st.number_input("Cholesterol", 0, 600, 200)
                fbs = st.selectbox("Fasting BS", [0, 1])
                ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
                hr = st.number_input("Max HR", 60, 202, 150)
            with col3:
                exang = st.selectbox("Exercise Angina", ["N", "Y"])
                oldpeak = st.number_input("Oldpeak", -3.0, 7.0, 0.0)
                st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
                
            submit = st.form_submit_button("Run Diagnostics", type="primary")

        if submit:
            input_df = pd.DataFrame([{
                "Age": age, "Sex": sex, "ChestPainType": cp, "RestingBP": bp,
                "Cholesterol": chol, "FastingBS": fbs, "RestingECG": ecg, "MaxHR": hr,
                "ExerciseAngina": exang, "Oldpeak": oldpeak, "ST_Slope": st_slope
            }])
            
            st.markdown("### Model Consensus")
            cols = st.columns(3)
            
            for (name, pipe), col in zip(pipelines.items(), cols):
                pred = pipe.predict(input_df)[0]
                status = "Disease" if pred == 1 else "Healthy"
                color = "inverse" if pred == 1 else "normal"
                col.metric(label=name, value=status, delta="Predicted", delta_color=color)

    # === TAB 3: TEST SET INSPECTION ===
    with tab_test:
        st.subheader("Audit Test Set Predictions")
        st.markdown("Compare the actual ground truth against all three models simultaneously.")
        
        idx = st.slider("Select Patient Index (Test Set)", 0, len(X_test)-1, 0)
        row_X = X_test.iloc[[idx]]
        actual = y_test.iloc[idx]
        
        st.markdown("---")

        cols = st.columns(4)

        cols[0].metric(
            label="Ground Truth", 
            value="Disease" if actual == 1 else "Healthy",
            delta="Actual Status",
            delta_color="off"
        )

        for (name, pipe), col in zip(pipelines.items(), cols[1:]):
            pred = pipe.predict(row_X)[0]
            
            if pred == actual:
                status, color = "Correct", "normal"
            else:
                status, color = "Incorrect", "inverse"
                
            col.metric(
                label=f"{name}", 
                value="Disease" if pred == 1 else "Healthy", 
                delta=status, 
                delta_color=color
            )
            
        st.markdown("---")
        st.markdown("**Patient Profile:**")
        st.dataframe(row_X, width='stretch')