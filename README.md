# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)
## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you, so do not share it. If you accidentally make it public, then you can create a new one with _Regenerate API Key_.

## Dataset Content

- **Source**: The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). This dataset provides an opportunity to create a predictive analytics solution in a real-world context, with a fictitious storyline for application in the workplace.
- **Description**: The dataset contains over 4,000 images of cherry leaves collected from Farmy & Foods’ cherry crop fields. These images are categorized into:
  - **Healthy Cherry Leaves**: Leaves without visible signs of mildew.
  - **Mildew-Affected Cherry Leaves**: Leaves showing signs of powdery mildew, a fungal disease that could compromise crop quality if not detected and treated early.

  As the cherry crop is a premium product for Farmy & Foods, ensuring the quality of these leaves is essential. Detecting powdery mildew is currently a manual process, making it time-intensive and challenging to scale across multiple farms.

## Business Requirements

Farmy & Foods currently relies on a manual inspection process to identify powdery mildew on cherry leaves. An employee inspects each tree for mildew by visually examining leaf samples, a process that takes about 30 minutes per tree. If mildew is detected, the employee then applies an anti-fungal compound in a one-minute application process. Given that the company has thousands of trees across various farms, this method is inefficient and unsustainable.

To address this, the IT team proposes a machine learning solution that can automatically detect mildew in leaf images, reducing inspection time and improving scalability. If successful, the company could extend this solution to other crops.

The business requirements are:
1. Conduct a study to identify and visually differentiate healthy cherry leaves from those affected by powdery mildew.
2. Create a predictive model that can accurately classify a cherry leaf as healthy or mildew-affected.


## Hypothesis and how to validate?

- **Hypothesis**: A machine learning model, particularly a Convolutional Neural Network (CNN), can accurately classify cherry leaf images as either healthy or mildew-affected, meeting a target accuracy of 97%.
  
- **Validation Approach**:
  1. **Training**: Train a CNN on labeled cherry leaf images to recognize visual features indicative of mildew.
  2. **Testing**: Evaluate model performance on a test set, ensuring that the classification accuracy meets the 97% threshold.
  3. **Performance Analysis**: Use accuracy scores, confusion matrices, and learning curves to validate the model’s effectiveness in differentiating the two classes.

## Rationale for Mapping Business Requirements to Data Visualizations and ML Tasks

Each business requirement will be supported by specific data visualizations and ML tasks as follows:

1. **Data Visualization**:
   - Use side-by-side visual comparisons, average images, and variability images to highlight differences between healthy and mildew-affected leaves, aiding both the exploratory analysis and training data preparation.
   
2. **ML Task**:
   - Develop a CNN model for binary classification to detect mildew presence. This model aligns with the business goal of predicting leaf health status, helping the client automate the inspection process.

3. **Dashboard**:
   - Implement an interactive dashboard to upload and predict leaf health status in real-time, addressing the client’s need for an easily accessible, quick inspection tool.

## ML Business Case

**Objective**: Automate the detection of powdery mildew on cherry leaves to save inspection time and ensure product quality.

**Model Selection**: A CNN model is chosen for its high accuracy in image classification, specifically in identifying visual patterns.

**Performance Metrics**: The model will be assessed based on accuracy, aiming for 97%, with further analysis using confusion matrices and accuracy curves to confirm consistent performance.

**Deployment Strategy**: The trained model will be integrated into a Streamlit dashboard, allowing end users to upload images for immediate classification results.
 
## Dashboard - Design Document

The dashboard will include the following pages and components:

### Page 1: Quick Project Summary

**Quick Project Summary**:
   - This project aims to address the detection of powdery mildew, a common fungal disease in cherry leaves. The tool helps farmers and agricultural professionals quickly and accurately determine the health of cherry leaves, saving time and resources.

**General Information**:
   - Powdery mildew is caused by fungi that thrive in warm and humid conditions.
   - It negatively impacts cherry production, leading to reduced fruit quality and yield.
   - Early detection is crucial to prevent the spread and reduce economic losses.

**Project Dataset**: 
   - The dataset contains over 4,000 images of cherry leaves, categorized into Healthy and Mildew-Affected.
   - The images were sourced from a publicly available dataset Cherry Leaves Dataset.

**Business Requirements**
   1. Provide a visual differentiation between healthy and mildew-affected leaves.
   2. Allow real-time predictions for new leaf images to determine their health status.

### Page 2: Visual Analysis of Cherry Leaves

This page addresses Business Requirement 1 by providing a visual analysis of the dataset.

**Components**:

1. **Checkbox: Difference Between Average and Variability Images**
   - Displays the average image of all healthy leaves vs. all mildew-affected leaves, showing key differences.

2. **Checkbox: Differences Between Average Healthy and Average Mildew-Affected Leaves**
   - Highlights the visual distinction between the two categories.

3. **Checkbox: Image Montage**
   - A grid montage displaying examples of healthy and mildew-affected leaves.
   
### Page 3: Mildew Detection:

This page addresses Business Requirement 2 by enabling live predictions for new images uploaded by the user.

**Components**:

1. **File Uploader Widget**
   - Allows users to upload one or more images of cherry leaves for mildew detection.

2. **Prediction Display**
   - Shows each uploaded image with a prediction statement, indicating whether it is Healthy or Mildew-Affected, along with the associated probability.

3. **Results Table**
   - Displays a table listing:
      - Image name
      - Prediction result (Healthy/Mildew-Affected)
      - Confidence score

4. **Download Button**
   - Allows the user to download the results table in CSV format.

### Page 4: Project Hypothesis and Validation

This page summarizes the hypotheses made during the project, conclusions drawn, and the methods used for validation.

**Blocks**:

1. **Hypothesis 1: Healthy and mildew-affected leaves have distinct visual features.**
   - Validated through average image comparisons and montages.

2. **Hypothesis 2: A machine learning model can accurately classify leaves as Healthy or Mildew-Affected.**
   - Validated using a Convolutional Neural Network (CNN) that achieved high accuracy on the test dataset.

### Page 5: ML Prediction Metrics

This page provides detailed insights into the model's performance.

**Components**:

1. **Label Frequencies**
   - A bar chart showing the distribution of Healthy and Mildew-Affected labels across the train, validation, and test sets.

2. **Model History: Accuracy and Losses**
   - Line plots visualizing the training and validation accuracy/loss over epochs.

3. **Model Evaluation Results**
   - A summary of metrics including:
      - Accuracy: Overall model accuracy on the test set.
      - Precision, Recall, and F1-Score: Performance metrics for each class.
      - Confusion Matrix: Visualized as a heatmap to show true positives, false positives, true negatives, and false negatives.

### Notes

- All pages are interactive, allowing users to explore different aspects of the project in depth.
- The dashboard is developed using Streamlit, ensuring a user-friendly interface and seamless navigation.

## Debugging and Model Evaluation

To ensure the reliability and accuracy of the mildew detection model, an extensive evaluation was conducted on the test set, focusing on both the overall performance and specific predictions for mildew-affected images.

### Key Steps in Debugging and Evaluation:

1. **Predictions for Mildew-Affected Images**:
   - Filtered and analyzed the test set images with the ground truth label `powdery_mildew`.
   - Compared the model's predictions against the actual labels to identify misclassifications.

2. **Overall Model Performance**:
   - Calculated key metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
   - Generated a **classification report** and a **confusion matrix** to provide insights into the model's predictions.

3. **Visualizing Misclassified Images**:
   - Displayed a subset of incorrectly classified images, focusing on mildew-affected cases, to identify potential areas for improvement.

### Results:

- **Accuracy**: Achieved an overall accuracy of `1.00`.
- **Precision and Recall**:
  - **Healthy Class**: Precision: `1.00`, Recall: `1.00`.
  - **Powdery Mildew Class**: Precision: `1.00`, Recall: `1.00`.
- **Confusion Matrix**:
  - Correctly classified "Healthy": N1 images.
  - Correctly classified "Powdery Mildew": N2 images.
  - Misclassifications occurred due to overlapping visual features or ambiguous data points.

  ![Confusion Matrix](jupyter_notebooks/outputs/v1/confusion_matrix.png)

### Learning Insights:
- Misclassified images revealed areas where the model struggled, such as unclear mildew patterns or similar texture/color between healthy and mildew-affected leaves.

### Next Steps:
- Expand the dataset to include more examples of challenging cases.
- Experiment with advanced model architectures and hyperparameter tuning.

## Unfixed Bugs

1. **Misclassifications**:
   - Certain mildew-affected leaves were misclassified as healthy during testing. This may be due to unclear mildew patterns or overlapping visual features.
   - Further refinement of the dataset and preprocessing steps could mitigate these issues.

2. **Edge Case Limitations**:
   - The model occasionally struggles with edge cases, such as partially visible mildew or poorly lit images. 

3. **Dashboard Performance**:
   - While functional, the Streamlit dashboard may experience delays or errors with large image files. Additional optimization is planned for future versions.

These bugs were not addressed due to time constraints but will be considered in future iterations of the project.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

1. **Pandas**: Used for data handling and image metadata management.

2. **NumPy**: This numerical computing library facilitates working with multi-dimensional arrays and matrices, which is vital for efficient image processing, statistical computations, and data manipulation.

3. **Matplotlib**: A versatile library for creating static, animated, and interactive visualizations in Python. Here, we used it primarily for visualizing image dimensions, generating plots to illustrate mean and variability of images, and for comparison images.

4. **Seaborn**: Built on top of matplotlib, Seaborn provides an interface for drawing attractive statistical graphics. It enhances visualizations and supports the generation of scatter plots, histograms, and other data visualizations to understand the image dimensions and label variability.

5. **TensorFlow/Keras**: Provides the framework to build and train the CNN model.

   - **tensorflow.keras.preprocessing.image**: A module within TensorFlow’s Keras API that supports image processing operations, such as loading, resizing, and converting images to arrays. This is crucial for preparing our image data as arrays, which are compatible with machine learning algorithms.

6. **Streamlit**: Supports the creation of an interactive dashboard for real-time predictions.

7. **Joblib**: Used for saving and loading large datasets, models, and other numerical data. Here, we saved our computed image shape embeddings as a .pkl file, making it easy to reuse across different parts of the project.

8. **random**: A library for generating random selections and orders, used here to help create a randomized image montage. This module provides flexibility in displaying a variety of images without manual selection.

9. **itertools**: A Python standard library for efficient looping and combining items. We utilized it here to manage the indices in the montage, ensuring that images align correctly in the grid layout.


## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The project introduction and context were adapted from the Code Institute’s project template.
- The deployment steps were referenced from [Heroku’s Python documentation](https://devcenter.heroku.com/articles/python-support).

### Media

- Cherry leaf images were sourced from the [Kaggle dataset](https://www.kaggle.com/codeinstitute/cherry-leaves) provided by Code Institute.
- Icons and visual elements in the dashboard were sourced from [Font Awesome](https://fontawesome.com/).

## Acknowledgements (optional)

- Thanks to Code Institute for providing the dataset and project framework, and to Kaggle for hosting the dataset. Special thanks to my mentors and peers for their guidance and support throughout the project.