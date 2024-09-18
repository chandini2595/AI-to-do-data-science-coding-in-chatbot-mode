# CMPE255_Assignment4

Use AI to do data science/coding in chatbot mode

Assignment 1:

Do data science (ideally deep learning but simpler one is fine)  using chatgpt code interpreter by picking a popular data set in kaggle website, upload and do various modules of data science - like my example . Export your chat transcript and submit it. 
Publish a medium article by selecting sections of the output 
Generate a nice report - in medium.com. use savegpt extension to export your work to pdf, import and insert any images from chatgpt (screenshot extension) 
submit the medium article
—------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Goal: The main goal is to build a convolutional neural network (CNN) to classify handwritten digits (0-9) using the MNIST dataset from Kaggle.
Tools: ChatGPT’s code interpreter, Python, TensorFlow, and Matplotlib for data visualization.

We will follow the CRISP-DM methodology (Cross Industry Standard Process for Data Mining) to guide our analysis of the Kaggle dataset for handwritten digit recognition using CNN. CRISP-DM consists of six main phases:
Business Understanding
Data Understanding
Data Preparation
Modeling
Evaluation
Deployment
Step 1: Business Understanding
The goal of this project is to develop a machine learning model using a Convolutional Neural Network (CNN) to automatically recognize handwritten digits from a dataset. This task is common in optical character recognition (OCR) applications, such as scanning handwritten documents, reading postal codes, and enabling digit-based user input on devices.
Step 2: Data Understanding
In the Data Understanding phase, we explore the data that will be used to build our model. The goal is to familiarize ourselves with the data, assess its quality, and determine if any issues need to be addressed before proceeding to modeling.
Key Objectives:
Initial Data Collection: Import and inspect the dataset provided to understand its format and contents.
Descriptive Statistics: Explore key characteristics of the dataset, such as size, distribution, and any missing values.

Visualization: Visualize samples of the data to gain an intuitive understanding of the patterns and challenges in the recognition task.
Quality Check: Identify any missing, corrupt, or irrelevant data that could impair model performance.

Step 3: Data Preparation
In the Data Preparation phase, we will get the dataset ready for model training. This includes pre-processing steps to ensure that the images are properly formatted for input into the CNN model. Key steps include:
Normalization: Normalize pixel values to a range between 0 and 1, as this often improves the performance of neural networks.
Reshaping: Ensure the images are in a format suitable for the CNN, typically with dimensions like (number of images, height, width, channels). For grayscale images, the number of channels will be 1.
Train-Test Split: Although this is a test dataset, typically, we might split data into training and validation sets. Since this dataset is often used as a test set in conjunction with the training set from the MNIST dataset, we will treat it as is unless otherwise required.
Data Augmentation (Optional): Techniques like rotating, flipping, or zooming into images can help the model generalize better by artificially increasing the dataset size.
Step 4: Modeling
CNNs are widely used for image classification tasks due to their ability to learn spatial hierarchies of features from input images. We'll build a CNN model using Keras, a popular deep learning library.
CNN Architecture Outline:
Input Layer: Takes in the 28x28x1 image.
Convolutional Layers: Apply multiple filters to extract features from the image.
Pooling Layers: Reduce the spatial dimensions (height and width) while retaining the most important information.
Flatten Layer: Flattens the output from the convolutional layers into a single vector.
Fully Connected Layers (Dense): Standard neural network layers to perform classification.
Output Layer: A softmax output layer with 10 units (one for each digit class 0-9).
Step 5: Model Evaluation
Once the CNN model is trained, the next crucial step is to evaluate its performance. In this step, we assess how well the model generalizes to unseen data and whether it meets the project's objectives.
Key Metrics for Model Evaluation:
Accuracy: Measures the percentage of correctly classified digits out of the total samples.
Confusion Matrix: Provides a detailed breakdown of classification results, showing where the model is confusing certain digits.
Precision, Recall, and F1-Score: These metrics help assess model performance for each digit class (0-9), providing insights into false positives and false negatives.
Loss and Accuracy Curves: These visualizations show how the model’s performance evolves over training epochs.
Step 6: Model Deployment
Once a model is trained and evaluated, deployment is the final step where the model is integrated into a production environment. The goal is to allow the model to make predictions in real-time or batch processes. Deploying a deep learning model like a CNN for handwritten digit recognition involves several considerations.
Here are common ways to deploy a machine learning model:
Deploying as a Web Service: The model can be served through a REST API, allowing applications to send image data and receive predictions.
Deploying as a Mobile/Embedded Application: Export the model to a format compatible with mobile or edge devices, like TensorFlow Lite.
Deploying to the Cloud: Use cloud services such as AWS, Google Cloud, or Azure to deploy the model at scale.

Chatgpt chat transcript: https://chatgpt.com/share/66eb3af2-e930-8004-af2c-12e57bba1a11
Colab implementation file:

