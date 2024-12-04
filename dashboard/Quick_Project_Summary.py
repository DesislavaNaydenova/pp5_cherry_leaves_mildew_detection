import streamlit as st
import matplotlib.pyplot as plt


def Quick_Project_Summary_body():

    st.write("## Quick Project Summary")

    st.info(
        f"**General Information**\n"
        f"* Malaria is a parasitic infection transmitted by the bite of infected female "
        f"Anopheles mosquitoes.\n")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/DesislavaNaydenova/pp5_cherry_leaves_mildew_detection/blame/main/README.md).")
    


def Quick_Project_Summary_body():
    """Displays the Quick Project Summary page for the Cherry Leaves Mildew Detection Dashboard."""

    st.write("### Quick Project Summary")
    
    # General Information
    st.info(
        f"**General Information**\n"
        f"* Powdery mildew is a fungal disease affecting various plants, including cherry trees. "
        f"It can significantly impact crop yield and quality.\n"
        f"* Early detection of powdery mildew is critical to minimize damage and prevent the spread of the disease.\n\n"
        f"**Project Dataset**\n"
        f"* The dataset contains images of cherry leaves categorized into two classes:\n"
        f"  - Healthy\n"
        f"  - Powdery Mildew\n"
        f"* These images are split into training, validation, and test datasets to develop and evaluate a machine learning model."
    )

    # Business Requirements
    st.success(
        f"The project addresses the following business requirements:\n"
        f"* 1 - Provide a study differentiating healthy cherry leaves from those infected with powdery mildew.\n"
        f"* 2 - Develop a predictive model to classify whether a given cherry leaf image is healthy or infected."
    )

    # Call to Action
    st.write(
        f"For further details, you can visit the "
        f"[GitHub repository](https://github.com/DesislavaNaydenova/pp5_cherry_leaves_mildew_detection) hosting the project"
        f" and **read** the "
        f"[Project README file](https://github.com/DesislavaNaydenova/pp5_cherry_leaves_mildew_detection/blob/main/README.md)."
    )
