Plant Disease Detection from Leaf Images
Overview
This project is a Plant Disease Detection System built using Python, PyTorch, OpenCV, and Streamlit. It enables users to upload a plant leaf image through a web interface and accurately detect the disease class present using a custom CNN or pretrained deep learning models. The project integrates computer vision, transfer learning, and interactive UI to provide a powerful tool for real-time plant disease classification.

Features
Image Upload Interface: Streamlit-based UI supporting PNG/JPG uploads for real-time prediction.

Disease Classification: Classifies plant leaf images into one of 38 disease classes using:

Custom CNN

ResNet50

EfficientNet

DenseNet

Model Selection: Choose between custom or pretrained models for prediction.

Grad-CAM Heatmap: Visual explanation showing the important regions of the image.

Training & Evaluation Pipeline: Reusable scripts for training custom and pretrained models with accuracy/loss visualization.

Test Set Evaluation: Reports test accuracy and loss on unseen data for each model.

Technologies Used
Python

PyTorch

torchvision

Streamlit

OpenCV

matplotlib

tqdm

Dataset
New Plant Diseases Dataset from Kaggle

Classes: 38 plant disease categories (including healthy leaves)

Dataset Structure:

train/: Training images categorized in class folders

valid/: Validation images categorized similarly

test_data/: Test images for final evaluation

Prerequisites
Ensure you have the following installed:

Python (>=3.9)

pip

Required Python libraries:

bash
Copy
Edit
pip install torch torchvision opencv-python matplotlib streamlit tqdm
Usage
Train the model:

Run custom_cnn_train.py for training the custom CNN.

Run pretrained_model_train.py for training ResNet, EfficientNet, or DenseNet.

Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Upload a leaf image.

Select the model for prediction.

View the predicted class and Grad-CAM visualization.

CNN Model Architecture
3 Convolutional layers with BatchNorm + ReLU + MaxPooling

2 Fully connected layers with Dropout

Input size: 128x128 RGB image

Output: 38 plant disease classes

Directory Structure
graphql
Copy
Edit
├── train/                        # Training data
├── valid/                        # Validation data
├── test_data/                    # Testing data
├── app.py                        # Streamlit web app
├── custom_cnn_train.py           # Custom CNN training script
├── pretrained_model_train.py     # Script to train ResNet, DenseNet, EfficientNet
├── gradcam_utils.py              # Grad-CAM explanation helper
├── best_custom_cnn_3.pth         # Trained custom model weights
Contribution
Feel free to contribute by forking the repository and submitting pull requests.

License
This project is licensed under the MIT License.

Author
Ramadevi N
