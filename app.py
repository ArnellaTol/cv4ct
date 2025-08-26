# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import hashlib

# # --- App Header ---
# st.title("🧠 CV4CT: Computer Vision for Sinus CT Scan Support")
# st.write("AI Assistant for detecting sinus tumors and identifying the affected side.")

# # --- Image preprocessing ---
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # --- Load Model 1: Tumor Presence ---
# @st.cache_resource
# def load_presence_model(path='resnet18_fold5.pth'):
#     model = models.resnet18(pretrained=False)
#     model.fc = nn.Linear(model.fc.in_features, 2)
#     model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # --- Load Model 2: Tumor Side ---
# @st.cache_resource
# def load_side_model(path='models_tumor_side/best_model_fold2.pt'):
#     model = models.resnet18(pretrained=False)
#     model.fc = nn.Linear(model.fc.in_features, 3)
#     model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
#     model.eval()
#     return model

# # --- Load models ---
# presence_model = load_presence_model()
# side_model = load_side_model()

# # --- Upload image ---
# uploaded_file = st.file_uploader("Upload a sinus CT scan", type=['jpg', 'png', 'jpeg'])

# if uploaded_file:
#     image = Image.open(uploaded_file).convert('RGB')

#     if st.button("Analyze"):
#         input_tensor = transform(image).unsqueeze(0)

#         with torch.no_grad():
#             tumor_logits = presence_model(input_tensor)
#             tumor_probs = torch.softmax(tumor_logits, dim=1).squeeze()
#             tumor_pred = int(torch.argmax(tumor_probs))
#             tumor_confidence = float(tumor_probs[tumor_pred])

#         st.subheader("Tumor Detection:")
#         st.progress(tumor_confidence)
#         if tumor_pred == 1:
#             st.error(f"Tumor detected with confidence: **{tumor_confidence:.1%}**")

#             with torch.no_grad():
#                 side_logits = side_model(input_tensor)
#                 side_probs = torch.softmax(side_logits, dim=1).squeeze()
#                 side_pred = int(torch.argmax(side_probs))
#                 side_confidence = float(side_probs[side_pred])

#             side_labels = {0: 'Left side', 1: 'Right side', 2: 'Both sides'}

#             st.subheader("Affected Side:")
#             st.progress(side_confidence)
#             st.warning(f"Affected: **{side_labels[side_pred]}** with confidence **{side_confidence:.1%}**")
#         else:
#             st.success(f"No tumor detected. Confidence: **{tumor_confidence:.1%}**")
    
#     st.image(image, caption='Uploaded CT Image', use_container_width=True)



import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# --- App Header ---
st.title("🧠 CV4CT: Computer Vision for Sinus CT Scan Support")
st.write("AI Assistant for detecting sinus tumors and identifying the affected side.")

# --- Utils ---
def preprocess(image: Image.Image) -> np.ndarray:
    """Resize to 224x224, convert to float32 [0,1], HWC->CHW, add batch dim."""
    image = image.resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)  # [1,3,224,224]
    return arr

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D logits."""
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return exps / np.sum(exps)

def run_onnx(session: ort.InferenceSession, inp: np.ndarray) -> np.ndarray:
    """Run ONNX session and return 1D probs (softmax over class logits)."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})
    logits = np.squeeze(outputs[0]).astype(np.float32)
    if logits.ndim > 1:
        logits = logits.reshape(-1)
    probs = softmax(logits)
    return probs

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def analyze_image(image: Image.Image):
    """Pipeline: preprocess -> tumor detection -> side detection."""
    input_arr = preprocess(image)

    # Tumor presence
    tumor_probs = run_onnx(presence_model, input_arr)
    tumor_pred = int(np.argmax(tumor_probs))
    tumor_confidence = float(tumor_probs[tumor_pred])

    st.subheader("Tumor Detection:")
    st.progress(clamp01(tumor_confidence))
    if tumor_pred == 1:
        st.error(f"Tumor detected with confidence: **{tumor_confidence:.1%}**")

        # Tumor side
        side_probs = run_onnx(side_model, input_arr)
        side_pred = int(np.argmax(side_probs))
        side_confidence = float(side_probs[side_pred])

        side_labels = {0: "Left side", 1: "Right side", 2: "Both sides"}

        st.subheader("Affected Side:")
        st.progress(clamp01(side_confidence))
        st.warning(
            f"Affected: **{side_labels[side_pred]}** "
            f"with confidence **{side_confidence:.1%}**"
        )
    else:
        st.success(f"No tumor detected. Confidence: **{tumor_confidence:.1%}**")

# --- Load models ---
@st.cache_resource
def load_presence_model() -> ort.InferenceSession:
    return ort.InferenceSession("presence_model.onnx", providers=["CPUExecutionProvider"])

@st.cache_resource
def load_side_model() -> ort.InferenceSession:
    return ort.InferenceSession("side_model.onnx", providers=["CPUExecutionProvider"])

presence_model = load_presence_model()
side_model = load_side_model()

# --- Tabs for user interaction ---
tab1, tab2 = st.tabs(["📂 Upload your file", "🖼️ Use sample images"])

with tab1:
    uploaded_file = st.file_uploader("Upload a sinus CT scan", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded CT Image", use_container_width=True)

        if st.button("Analyze", key="analyze_uploaded"):
            analyze_image(image)

with tab2:
    test_dir = "test_for_users"
    if os.path.exists(test_dir):
        sample_images = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if sample_images:
            choice = st.selectbox("Choose a sample CT image:", sample_images)
            image_path = os.path.join(test_dir, choice)
            image = Image.open(image_path).convert("RGB")
            st.image(image, caption=f"Sample: {choice}", use_container_width=True)

            if st.button("Analyze", key="analyze_sample"):
                analyze_image(image)
        else:
            st.info("⚠️ No sample images found in `test_for_users/` folder.")
    else:
        st.info("⚠️ Folder `test_for_users/` not found. Please create it and add images.")
