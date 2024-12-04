import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
import joblib

def load_evaluation_metrics(version="v1"):
    # Path to the model evaluation pickle file
    evaluation_path = f"jupyter_notebooks/outputs/{version}/model_evaluation.pkl"
    
    # Load the evaluation data from the pickle file
    try:
        evaluation = joblib.load(evaluation_path)
        return evaluation
    except FileNotFoundError:
        st.error(f"Evaluation file not found at {evaluation_path}")
        return None

def ML_Prediction_Metrics_metrics():

    version = 'v1'

    st.write("## ML Prediction Metrics")

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    # Display labels distribution image
    labels_distribution = plt.imread(f"jupyter_notebooks/outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation, and Test Sets')
    st.write("---")

    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"jupyter_notebooks/outputs/{version}/model_accuracy_plot.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"jupyter_notebooks/outputs/{version}/model_loss_plot.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalized Performance on Test Set")
    test_metrics = load_evaluation_metrics(version)
    st.dataframe(pd.DataFrame([test_metrics], index=["Test Set"]))