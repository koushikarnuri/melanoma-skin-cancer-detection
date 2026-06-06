import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --------------------
# CONFIG
# --------------------
st.set_page_config(
    page_title="Melanoma AI System",
    page_icon="🩺",
    layout="wide"
)

# --------------------
# LOAD MODEL
# --------------------
@st.cache_resource
def get_model():
    return load_model("melanoma_model.keras")

model = get_model()

# --------------------
# SIDEBAR
# --------------------
st.sidebar.title("🩺 Melanoma AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Predict",
        "📊 Model Performance",
        "📂 Dataset Info",
        "👨‍💻 About"
    ]
)

# --------------------
# DASHBOARD
# --------------------
if page == "🏠 Dashboard":

    st.title("🩺 Melanoma Skin Cancer Detection")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Dataset Images","10,682")
    col2.metric("Classes","2")
    col3.metric("Accuracy","77.48%")
    col4.metric("Framework","TensorFlow")

    st.divider()

    st.markdown("""
    ## 📌 Project Overview

    This system uses Deep Learning to classify skin lesions as:

    - Melanoma
    - Not Melanoma

    The model was trained using dermoscopic skin lesion images.
    """)

# --------------------
# PREDICT
# --------------------
elif page == "🔍 Predict":

    st.title("🔍 Skin Lesion Prediction")

    uploaded = st.file_uploader(
        "Upload Skin Lesion Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        col1,col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        img = image.resize((128,128))
        img = np.array(img)/255.0
        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img)[0][0]

        with col2:

            st.subheader("Prediction Result")

            if prediction < 0.5:

                confidence = (1-prediction)*100

                st.error("⚠️ Melanoma Detected")

                st.progress(int(confidence))

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            else:

                confidence = prediction*100

                st.success("✅ Not Melanoma")

                st.progress(int(confidence))

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

# --------------------
# MODEL PERFORMANCE
# --------------------
elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.metric("Training Accuracy","73.36%")
    st.metric("Validation Accuracy","77.48%")

    st.markdown("""
    ### Performance Summary

    - CNN Based Model
    - Binary Classification
    - Image Size: 128x128
    - Optimizer: Adam
    - Loss: Binary Crossentropy
    """)

# --------------------
# DATASET INFO
# --------------------
elif page == "📂 Dataset Info":

    st.title("📂 Dataset Information")

    st.markdown("""
    ### Dataset Structure

    Dataset/
    ├── Melanoma
    └── NotMelanoma

    ### Classes

    - Melanoma
    - Not Melanoma

    ### Source

    HAM10000 inspired skin lesion dataset.
    """)

# --------------------
# ABOUT
# --------------------
elif page == "👨‍💻 About":

    st.title("👨‍💻 About Project")

    st.markdown("""
    ### Melanoma Skin Cancer Detection

    Developed using:

    - Python
    - TensorFlow
    - Streamlit
    - NumPy
    - PIL

    ### Author

    Koushik Arnuri

    ### Disclaimer

    This application is for educational and research purposes only.
    It should not replace professional medical diagnosis.
    """)
