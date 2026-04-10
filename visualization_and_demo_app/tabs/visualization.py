import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

def render_visualization_tab(df):
    st.title("Heart Disease Clinical Dashboard")
    st.markdown("Explore the dataset to understand how different demographics and clinical markers influence heart disease prevalence.")

    # --- 1. GLOBAL INTERACTIVE FILTERS ---
    with st.expander("Filter", expanded=True):
        col_f1, col_f2 = st.columns(2)
        
        min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
        selected_age = col_f1.slider("Select Age Range:", min_age, max_age, (min_age, max_age))
        
        selected_sex = col_f2.multiselect("Select Sex (Leave empty for ALL):", options=df['Sex'].unique())

    plot_df = df[(df['Age'] >= selected_age[0]) & (df['Age'] <= selected_age[1])].copy()
    if selected_sex:
        plot_df = plot_df[plot_df['Sex'].isin(selected_sex)]

    if plot_df.empty:
        st.warning("No patients match the selected criteria.")
        return

    plot_df['Diagnosis'] = plot_df['HeartDisease'].map({0: 'Healthy', 1: 'Disease'})
    color_map = {'Healthy': '#1f77b4', 'Disease': '#d62728'} # Blue vs Red

    # --- 2. KEY METRICS ---
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    total_patients = len(plot_df)
    disease_cases = len(plot_df[plot_df['HeartDisease'] == 1])
    disease_rate = (disease_cases / total_patients) * 100 if total_patients > 0 else 0
    
    c1.metric("Total Patients", f"{total_patients}")
    c2.metric("Heart Disease Cases", f"{disease_cases}")
    c3.metric("Disease Prevalence", f"{disease_rate:.1f}%")

    st.markdown("---")

    # --- ROW 1: DEMOGRAPHICS & VITALS ---
    st.subheader("1. Demographics & Vitals")
    col1, col2 = st.columns(2)

    with col1:
        fig_age = px.violin(
            plot_df, x="Diagnosis", y="Age", color="Diagnosis",
            title="Age Distribution Profile",
            box=True, points="all", 
            color_discrete_map=color_map
        )
        st.plotly_chart(fig_age, width='stretch')

    with col2:
        fig_hr = px.box(
            plot_df, x="Diagnosis", y="MaxHR", color="Diagnosis",
            title="Max Heart Rate Achieved Profile",
            points="all", # Overlays the individual patient data points
            color_discrete_map=color_map
        )
        fig_hr.update_layout(annotations=[dict(
            text="Notice that lower Max Heart Rates are highly concentrated in the Disease group.",
            x=0.5, y=1.1, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color="gray")
        )])
        st.plotly_chart(fig_hr, width='stretch')

    # --- ROW 2: CATEGORICAL CLINICAL SYMPTOMS ---
    st.markdown("---")
    st.subheader("2. Clinical Symptoms")
    col3, col4 = st.columns(2)

    with col3:
        fig_cp = px.histogram(
            plot_df, x="ChestPainType", color="Diagnosis", barmode="group",
            title="Chest Pain Type Analysis",
            color_discrete_map=color_map,
            category_orders={"ChestPainType": ["ATA", "NAP", "ASY", "TA"]}
        )
        st.plotly_chart(fig_cp, width='stretch')

    with col4:
        fig_ea = px.histogram(
            plot_df, x="ExerciseAngina", color="Diagnosis", barmode="group",
            title="Exercise-Induced Angina Impact",
            color_discrete_map=color_map
        )
        st.plotly_chart(fig_ea, width='stretch')

    # --- ROW 3: ADVANCED MARKERS & CORRELATION ---
    st.markdown("---")
    st.subheader("3. Advanced Biomarkers & Correlation")
    col5, col6 = st.columns(2)

    with col5:
        fig_scatter = px.scatter(
            plot_df, x="Age", y="MaxHR", color="Diagnosis", 
            title="Maximum Heart Rate vs. Age Decline",
            color_discrete_map=color_map, opacity=0.8,
            trendline="ols" 
        )
        st.plotly_chart(fig_scatter, width='stretch')

    with col6:
        st.markdown("**Feature Correlation Heatmap**")
        numeric_cols = plot_df.select_dtypes(include=[np.number])
        corr_matrix = numeric_cols.corr()
        
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            aspect="auto",
            color_continuous_scale="RdBu_r", 
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_corr, width='stretch')