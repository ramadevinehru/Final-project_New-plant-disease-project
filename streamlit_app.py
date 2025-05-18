import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# ------------------------ Define Model ------------------------
class AdvancedPlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedPlantDiseaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------ Class Names ------------------------
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Cleaned class names for UI display
clean_class_names = [name.replace("___", ": ").replace("_", " ") for name in class_names]

# ------------------------ Transforms ------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ------------------------ Load Model ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedPlantDiseaseCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("best_custom_cnn_3.pth", map_location=device))
model.eval()

# ------------------------ Streamlit UI ------------------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect the plant disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        _, pred_label = torch.max(output, 1)

    st.markdown(f"### 🧠 Prediction: **{clean_class_names[pred_label.item()]}**")





# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2

# # ---------------------- Model Definition ----------------------
# class AdvancedPlantDiseaseCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(AdvancedPlantDiseaseCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#         )

#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 16 * 16, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# # ---------------------- Class Names ----------------------
# class_names = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#     'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
#     'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
#     'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
#     'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
#     'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
#     'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
#     'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
#     'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
#     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#     'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
# ]

# # ---------------------- Load Trained Model ----------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AdvancedPlantDiseaseCNN(num_classes=len(class_names)).to(device)
# model.load_state_dict(torch.load("best_custom_cnn_3.pth", map_location=device))
# model.eval()

# # ---------------------- Image Transform ----------------------
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])

# # ---------------------- Grad-CAM Generator ----------------------
# def generate_gradcam(image_tensor, model, class_idx):
#     gradients = []
#     activations = []

#     def backward_hook(module, grad_in, grad_out):
#         gradients.append(grad_out[0])

#     def forward_hook(module, input, output):
#         activations.append(output)

#     hook_f = model.features[-3].register_forward_hook(forward_hook)
#     hook_b = model.features[-3].register_backward_hook(backward_hook)

#     image_tensor = image_tensor.unsqueeze(0).to(device).requires_grad_()
#     output = model(image_tensor)
#     model.zero_grad()
#     output[0, class_idx].backward()

#     pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
#     activations_map = activations[0].detach()[0]

#     for i in range(len(pooled_gradients)):
#         activations_map[i] *= pooled_gradients[i]

#     heatmap = torch.mean(activations_map, dim=0).cpu().numpy()
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap + 1e-8)

#     hook_f.remove()
#     hook_b.remove()

#     return heatmap

# # ---------------------- Show Grad-CAM ----------------------
# def show_gradcam(image_tensor, class_idx):
#     heatmap = generate_gradcam(image_tensor, model, class_idx)
#     img = image_tensor.permute(1, 2, 0).cpu().numpy()
#     img = np.clip(img, 0, 1)

#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     superimposed_img = cv2.addWeighted(np.uint8(img * 255), 0.6, heatmap_color, 0.4, 0)

#     return superimposed_img

# # ---------------------- Streamlit UI ----------------------
# st.title("🌿 Plant Disease Detection")
# st.write("Upload an image of a leaf to classify its disease and visualize the focus area.")

# uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Uploaded Image', use_column_width=True)

#     input_tensor = transform(image)

#     with torch.no_grad():
#         output = model(input_tensor.unsqueeze(0).to(device))
#         probabilities = torch.softmax(output, dim=1)
#         confidence, predicted_class = torch.max(probabilities, 1)
#         predicted_label = class_names[predicted_class.item()]
#         confidence_pct = confidence.item() * 100

#     st.markdown(f"### 🔍 Prediction: **{predicted_label}**")
#     st.markdown(f"#### ✅ Confidence: `{confidence_pct:.2f}%`")

#     # Grad-CAM visualization
#     st.subheader("🔬 Model Focus Area (Grad-CAM)")
#     gradcam_result = show_gradcam(input_tensor, predicted_class.item())
#     st.image(gradcam_result, caption="Grad-CAM Heatmap", use_column_width=True)

