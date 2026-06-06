```python
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="Melanoma Detection", layout="wide")

st.title("🩺 Melanoma Skin Cancer Detection")

MODEL_PATH = "model/densenet_model.keras"
CATEGORIES = ["Melanoma", "NotMelanoma"]

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

# ==================================================
# CHECK MODEL EXISTS
# ==================================================
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        "Upload your trained densenet_model.keras file "
        "inside the model folder."
    )
    st.stop()

model = get_model()

# ==================================================
# IMAGE UPLOAD
# ==================================================
uploaded_file = st.file_uploader(
    "Upload Skin Lesion Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Image", width=300)

    image = load_img(uploaded_file, target_size=(32, 32))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction) * 100)

    st.success(f"Prediction: {CATEGORIES[class_index]}")
    st.info(f"Confidence: {confidence:.2f}%")
```
