import streamlit as st
import os
import itertools
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

def Visual_Analysis_of_Cherry_Leaves_body():
    """Displays the visual analysis of cherry leaves."""

    st.write("## Visual Analysis of Cherry Leaves")

    st.info(
        f"* The client is interested in studying the visual differences between "
        f"healthy and powdery mildew-infected cherry leaves."
    )
    
    version = 'v1'

    # Option: Difference between average and variability images
    if st.checkbox("Difference between Average and Variability Images"):
        avg_healthy = plt.imread(f"jupyter_notebooks/outputs/{version}/avg_var_Healthy.png")
        avg_powdery_mildew = plt.imread(f"jupyter_notebooks/outputs/{version}/avg_var_Powdery_Mildew.png")

        st.warning(
            f"* The average and variability images do not reveal clear, intuitive "
            f"patterns to differentiate between healthy and powdery mildew-infected leaves. "
            f"A slight difference in color pigment can be observed in the average images."
        )
        
        st.image(avg_healthy, caption="Healthy Leaves - Average and Variability")
        st.image(avg_powdery_mildew, caption="Powdery Mildew Leaves - Average and Variability")
        st.write("---")

    # Option: Difference between average healthy and mildew leaves
    if st.checkbox("Difference Between Average Healthy and Powdery Mildew Images"):
        diff_between_avgs = plt.imread(f"jupyter_notebooks/outputs/{version}/avg_diff.png")

        st.warning(
            f"* The difference between the average images of healthy and mildew-infected leaves "
            f"does not reveal distinct patterns for easy visual differentiation."
        )
        st.image(diff_between_avgs, caption="Difference Between Average Images")
        st.write("---")

    # Option: Image Montage
    if st.checkbox("Image Montage"):
        st.write("* To refresh the montage, click on the 'Create Montage' button")
        my_data_dir = "inputs/cherry_leaves_dataset/cherry-leaves/test"
        labels = os.listdir(my_data_dir)
        label_to_display = st.selectbox(label="Select Label", options=labels, index=0)
        
        if st.button("Create Montage"):
            image_montage(dir_path=my_data_dir, label_to_display=label_to_display, nrows=3, ncols=3, figsize=(10, 10))
        st.write("---")


def image_montage(dir_path, label_to_display, nrows=3, ncols=3, figsize=(15, 10)):
    """Creates and displays a montage of images for the given label."""
    sns.set_style("white")
    labels = os.listdir(dir_path)

    # Ensure the specified label exists
    if label_to_display in labels:
        images_list = os.listdir(os.path.join(dir_path, label_to_display))

        # Check if montage space is sufficient
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.error(
                f"Decrease the number of rows or columns. "
                f"There are {len(images_list)} images in the subset, "
                f"but the requested montage size is {nrows * ncols}."
            )
            return

        # Generate montage
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, img_name in enumerate(img_idx):
            img = imread(os.path.join(dir_path, label_to_display, img_name))
            img_shape = img.shape
            ax = axes[i // ncols, i % ncols]
            ax.imshow(img)
            ax.set_title(f"Width: {img_shape[1]}px, Height: {img_shape[0]}px")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig=fig)
    else:
        st.error("The specified label does not exist.")
        st.write(f"Available labels are: {labels}")
