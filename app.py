import streamlit as st
from dashboard.multipage import MultiPage

# load pages scripts
from dashboard.Quick_Project_Summary import Quick_Project_Summary_body
from dashboard.Visual_Analysis_of_Cherry_Leaves import Visual_Analysis_of_Cherry_Leaves_body
from dashboard.Mildew_Detection import Mildew_Detection_body
from dashboard.Project_Hypothesis_Validation import Project_Hypothesis_Validation_body
from dashboard.ML_Prediction_Metrics import ML_Prediction_Metrics_metrics

app = MultiPage(app_name= " Mildew Detection in Cherry Leaves") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", Quick_Project_Summary_body)
app.add_page("Visual Analysis of Cherry Leaves", Visual_Analysis_of_Cherry_Leaves_body)
app.add_page("Mildew Detection", Mildew_Detection_body)
app.add_page("Project Hypothesis", Project_Hypothesis_Validation_body)
app.add_page("ML Performance Metrics", ML_Prediction_Metrics_metrics)

app.run() # Run the  app