import streamlit as st
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from tabs.visualization import render_visualization_tab
from tabs.training import render_training_tab

st.set_page_config(page_title="Heart Health Analytics", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    def robust_read_csv(filename):
        paths = [filename, f"../data/raw/{filename}", f"data/raw/{filename}"]
        for p in paths:
            if os.path.exists(p): return pd.read_csv(p)
        return None

    df = robust_read_csv("heart.csv")
    if df is None:
        st.error("Data not found! Please check your dataset folder.")
    return df

df = load_data()

if df is not None:
    tab1, tab2 = st.tabs(["Visualization", "Training & Inference"])
    
    with tab1:
        render_visualization_tab(df)
        
    with tab2:
        render_training_tab(df)