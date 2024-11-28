import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
# from src.machine_learning.evaluate_clf import load_test_evaluation


def ML_Prediction_Metrics_metrics():
    version = 'v1'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")