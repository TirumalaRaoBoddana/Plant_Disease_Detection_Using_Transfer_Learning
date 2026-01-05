import streamlit as st
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
@st.cache_resource
def load_data():
    model = load_model("final_resnet50_model.keras")

    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)

    # reverse mapping: index → class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class
def preprocess_image(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img_array = np.array(img)

    # remove alpha channel if present
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
def predict_disease(img, model, idx_to_class):
    processed_img = preprocess_image(img)
    preds = model.predict(processed_img)[0]

    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]

    predicted_class = idx_to_class[pred_idx]
    return predicted_class, confidence
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌱 Plant Disease Detection System")
st.write("Upload a leaf image to detect the disease")

model, idx_to_class = load_data()

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Disease"):
        disease, confidence = predict_disease(img, model, idx_to_class)

        st.markdown(
            f"""
            ### 🦠 Predicted Disease: **{disease}**
            ### 📊 Confidence: **{confidence * 100:.2f}%**
            """
        )

        if confidence < 0.6:
            st.warning("⚠️ Low confidence prediction. Image may be out of training domain.")
