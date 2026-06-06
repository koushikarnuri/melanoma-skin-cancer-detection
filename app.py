import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(page_title="Melanoma Skin Cancer Detection")

st.title("🩺 Melanoma Skin Cancer Detection")

# Load Model
@st.cache_resource
def load_my_model():
    return load_model("melanoma_model.keras")

model = load_my_model()

st.write("Upload a skin lesion image to predict Melanoma or Not Melanoma.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((128, 128))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    confidence = float(prediction)

    if confidence < 0.5:
        st.success("✅ Prediction: Melanoma")
        st.write(f"Confidence: {(1-confidence)*100:.2f}%")
    else:
        st.info("🟢 Prediction: Not Melanoma")
        st.write(f"Confidence: {confidence*100:.2f}%")
