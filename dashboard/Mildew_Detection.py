
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# Import your model functions 
from src.machine_learning.model import load_model_and_predict
from src.machine_learning.utils import resize_input_image, plot_predictions_probabilities


def Mildew_Detection_body():
    st.title("Cherry Leaf Mildew Detection")

    st.info(
        f"* This tool allows you to detect if a cherry leaf has mildew. "
        f"Upload an image of a cherry leaf, and the model will predict if mildew is present."
    )

    st.write(
        f"Upload an image of a cherry leaf to detect mildew presence."
    )

    try:
        # File uploader to accept cherry leaf image
        uploaded_image = st.file_uploader('Upload Cherry Leaf Image', type=['jpg','png'])
   
        if uploaded_image is not None:
            # Load and display the image
            img_pil = Image.open(uploaded_image)
            st.image(img_pil, caption="Uploaded Cherry Leaf Image", use_column_width=True)

            # Convert the image to a numpy array
            img_array = np.array(img_pil)
            st.write(f"Image dimensions: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            # Ensure the image is in RGB format
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")

            # Resize image for model input
            version = 'v1'  # Specify model version (can be dynamic if you have multiple models)
            resized_img = resize_input_image(img=img_pil, version=version)

            # Predict mildew presence
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)

            # Show the prediction results
            st.write(f"Prediction Result: **{'Mildew Detected' if pred_class == 1 else 'No Mildew Detected'}**")
            st.write(f"Prediction Probability: {pred_proba:.2f}")

            # Plot prediction probabilities (optional)
            plot_predictions_probabilities(pred_proba, pred_class)

            # Optionally, display additional information or results
            if pred_class == 1:
                st.warning("The image is predicted to have mildew.")
            else:
                st.success("The image is predicted to have no mildew.")
        
        else:
            st.write("Upload a cherry leaf image to begin the detection process.")

    except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")