import tensorflow as tf
import numpy as np

# Load the model
def load_model(version='v1'):
    model_path = f'jupyter_notebooks/outputs/{version}/mildew_detector_model.h5'  # Update path to your model
    model = tf.keras.models.load_model(model_path)
    return model

# Make prediction
def load_model_and_predict(image, version='v1'):
    model = load_model(version)
    # Preprocess the image
    image = np.expand_dims(image, axis=0) 
    predictions = model.predict(image)  # This gives an array of probabilities
    
    pred_proba = predictions[0][0]  # The the first value of probabilities for the input image
    pred_class = int(pred_proba > 0.5)
    return pred_proba, pred_class
