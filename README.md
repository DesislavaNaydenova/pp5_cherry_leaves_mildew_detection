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
 
## Dashboard Design

The dashboard will include the following pages and components:

1. **Project Summary Page**:
   - Displays the dataset summary, business objectives, and project goals.
   
2. **Visual Analysis Page**:
   - Shows visual differentiation findings, including average images and montages for healthy and mildew-affected leaves.
   
3. **Live Prediction Page**:
   - Contains an upload feature allowing users to upload multiple images for real-time mildew detection.
   - Provides prediction results, including classification probabilities.
   
4. **Technical Performance Page**:
   - Shows model performance metrics, such as accuracy scores and learning curves.

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

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
2. **NumPy**: Utilized for efficient handling of image arrays and transformations.
3. **Matplotlib & Seaborn**: Used to visualize data distributions, such as healthy vs. mildew-affected leaf counts.
4. **TensorFlow/Keras**: Provides the framework to build and train the CNN model.
5. **Streamlit**: Supports the creation of an interactive dashboard for real-time predictions.



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