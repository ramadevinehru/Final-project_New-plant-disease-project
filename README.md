
# Plant Disease Detection from Leaf Images

## Overview  
This project is a **Plant Disease Detection System** built using **Python, PyTorch, OpenCV**, and **Streamlit**. It allows users to upload a plant leaf image through a web interface and accurately detect the disease class using either a **custom CNN** or **pretrained deep learning models**.  

The system combines **computer vision**, **transfer learning**, and an **interactive UI** to offer a powerful tool for real-time plant disease classification.

---

## Features

- **Image Upload Interface**  
  Streamlit-based UI supporting PNG/JPG uploads for real-time prediction.

- **Disease Classification**  
  Classifies plant leaf images into one of **38 disease classes** using:
  - ✅ Custom CNN  
  - ✅ ResNet50  
  - ✅ EfficientNet  
  - ✅ DenseNet  

- **Model Selection**  
  Choose between custom or pretrained models for prediction.

- **Grad-CAM Heatmap**  
  Visual explanation highlighting important regions in the leaf image.

- **Training & Evaluation Pipeline**  
  Reusable scripts for training with accuracy/loss visualization.

- **Test Set Evaluation**  
  Reports test accuracy and loss on unseen data for each model.

---

## 🛠 Technologies Used

- Python  
- PyTorch  
- torchvision  
- Streamlit  
- OpenCV  
- matplotlib  
- tqdm

---

## Dataset

**New Plant Diseases Dataset** from [Kaggle](https://www.kaggle.com/).  
- **Classes**: 38 plant disease categories (including healthy leaves)  
- **Structure**:
  ```
  train/       # Training images categorized in class folders  
  valid/       # Validation images categorized similarly  
  test_data/   # Test images for final evaluation
  ```

---

## Prerequisites

Make sure you have the following installed:

- Python >= 3.9  
- pip

**Install required Python libraries:**
```bash
pip install torch torchvision opencv-python matplotlib streamlit tqdm
```

---

## 🚦 Usage

### Train the Model:
```bash
# Train the custom CNN
python custom_cnn_train.py

# Train pretrained models (ResNet, EfficientNet, DenseNet)
python pretrained_model_train.py
```

### Run the Streamlit App:
```bash
streamlit run app.py
```

- Upload a leaf image  
- Select the model for prediction  
- View the predicted class and Grad-CAM visualization

---

## CNN Model Architecture

- 3 Convolutional Layers:
  - BatchNorm + ReLU + MaxPooling  
- 2 Fully Connected Layers:
  - With Dropout  
- **Input**: 128×128 RGB image  
- **Output**: 38 plant disease classes

---

## Directory Structure

```plaintext
├── train/                        # Training data
├── valid/                        # Validation data
├── test_data/                    # Testing data
├── app.py                        # Streamlit web app
├── custom_cnn_train.py           # Custom CNN training script
├── pretrained_model_train.py     # Script to train ResNet, DenseNet, EfficientNet
├── gradcam_utils.py              # Grad-CAM explanation helper
├── best_custom_cnn_3.pth         # Trained custom model weights
```

---

## Contribution

Contributions are welcome!  
Feel free to **fork the repository** and submit a **pull request** with improvements.

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Ramadevi N**
