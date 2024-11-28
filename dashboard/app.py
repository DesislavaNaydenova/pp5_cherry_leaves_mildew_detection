import streamlit as st
from multipage import MultiPage

# load pages scripts
from pages.Quick_Project_Summary import Quick_Project_Summary_body
from pages.Cells_Visualizer import Cells_Visualizer_body
from pages.Live_Prediction import Live_Prediction_body
from pages.Project_Hypothesis_Validation import Project_Hypothesis_Validation_body
from pages.ML_Prediction_Metrics import ML_Prediction_Metrics_metrics

app = MultiPage(app_name= "Malaria Detector") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", Quick_Project_Summary_body)
app.add_page("Cells Visualizer", Cells_Visualizer_body)
app.add_page("Malaria Detection", Live_Prediction_body)
app.add_page("Project Hypothesis", Project_Hypothesis_Validation_body)
app.add_page("ML Performance Metrics", ML_Prediction_Metrics_metrics)

app.run() # Run the  app