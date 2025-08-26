import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import hashlib

# --- App Header ---
st.title("🧠 CV4CT: Computer Vision for Sinus CT Scan Support")
st.write("AI Assistant for detecting sinus tumors and identifying the affected side.")

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Load Model 1: Tumor Presence ---
@st.cache_resource
def load_presence_model(path='resnet18_fold5.pth'):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Load Model 2: Tumor Side ---
@st.cache_resource
def load_side_model(path='models_tumor_side/best_model_fold2.pt'):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Load models ---
presence_model = load_presence_model()
side_model = load_side_model()

# --- Upload image ---
uploaded_file = st.file_uploader("Upload a sinus CT scan", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    if st.button("Analyze"):
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            tumor_logits = presence_model(input_tensor)
            tumor_probs = torch.softmax(tumor_logits, dim=1).squeeze()
            tumor_pred = int(torch.argmax(tumor_probs))
            tumor_confidence = float(tumor_probs[tumor_pred])

        st.subheader("Tumor Detection:")
        st.progress(tumor_confidence)
        if tumor_pred == 1:
            st.error(f"Tumor detected with confidence: **{tumor_confidence:.1%}**")

            with torch.no_grad():
                side_logits = side_model(input_tensor)
                side_probs = torch.softmax(side_logits, dim=1).squeeze()
                side_pred = int(torch.argmax(side_probs))
                side_confidence = float(side_probs[side_pred])

            side_labels = {0: 'Left side', 1: 'Right side', 2: 'Both sides'}

            st.subheader("Affected Side:")
            st.progress(side_confidence)
            st.warning(f"Affected: **{side_labels[side_pred]}** with confidence **{side_confidence:.1%}**")
        else:
            st.success(f"No tumor detected. Confidence: **{tumor_confidence:.1%}**")
    
    st.image(image, caption='Uploaded CT Image', use_container_width=True)