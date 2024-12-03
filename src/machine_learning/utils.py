from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def resize_input_image(img, version='v1'):
        # Ensure image has 3 channels
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        raise ValueError("Input image must be RGB or RGBA.")

    # Resize to model's expected input size
    target_size = (256, 256)  
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

def plot_predictions_probabilities(pred_proba, pred_class):
    labels = ['No Mildew', 'Mildew']  # Adjust based on your model's labels
    fig, ax = plt.subplots()
    ax.bar(labels, pred_proba)
    ax.set_title(f'Prediction: {labels[pred_class]}')
    ax.set_ylabel('Probability')
    plt.show()