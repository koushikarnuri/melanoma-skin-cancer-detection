import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Melanoma AI System",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    return load_model("melanoma_model.keras")

model = load_my_model()

# =========================
# SIDEBAR
# =========================
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

# =========================
# DASHBOARD
# =========================
if page == "🏠 Dashboard":

    st.title("🩺 Melanoma Skin Cancer Detection")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Images", "10,682")
    col2.metric("Classes", "2")
    col3.metric("Accuracy", "77.48%")
    col4.metric("Framework", "TensorFlow")

    st.divider()

    st.markdown("""
    ## 📌 Project Overview

    This AI system helps classify skin lesion images into:

    - Melanoma
    - Not Melanoma

    Upload a dermoscopic image and receive an instant prediction.
    """)

# =========================
# PREDICT PAGE
# =========================
elif page == "🔍 Predict":

    st.title("🩺 Skin Cancer Screening")

    st.markdown("### 👤 Patient Information")

    c1, c2 = st.columns(2)

    with c1:
        patient_name = st.text_input(
            "Patient Name",
            placeholder="Enter patient name"
        )

        patient_age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=25
        )

    with c2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

        patient_id = st.text_input(
            "Patient ID",
            placeholder="Optional"
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "📤 Upload Skin Lesion Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.success(
            f"Hi {patient_name}, thank you for uploading your skin lesion image."
        )

        col1, col2 = st.columns([1, 1])

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        # FIXED: Preprocessing with 224x224 for DenseNet121
        img = image.resize((224, 224))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Get prediction
        prediction = float(model.predict(img, verbose=0)[0][0])

        # Debug: Show raw prediction
        st.write(f"**Raw Prediction Score:** {prediction:.4f}")

        with col2:

            st.subheader("🧾 Medical Screening Report")

            st.write(f"**Patient Name:** {patient_name}")
            st.write(f"**Patient ID:** {patient_id}")
            st.write(f"**Age:** {patient_age}")
            st.write(f"**Gender:** {gender}")

            st.divider()

            # Prediction logic
            if prediction > 0.5:

                confidence = prediction * 100

                st.error("⚠️ MELANOMA DETECTED")

                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%"
                )

                st.warning("""
                Recommendation:

                • Consult a dermatologist immediately.
                • Perform additional clinical examination.
                • Consider biopsy confirmation.
                """)

            else:

                confidence = (1 - prediction) * 100

                st.success("✅ NOT MELANOMA")

                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%"
                )

                st.info("""
                Recommendation:

                • No melanoma detected by the model.
                • Continue regular skin monitoring.
                • Consult a dermatologist if symptoms persist.
                """)

        st.divider()

        st.subheader("📋 Final Result")

        if prediction > 0.5:

            st.error(
                f"""
                Patient: {patient_name}

                Result: Melanoma Detected

                Confidence: {confidence:.2f}%
                """
            )

        else:

            st.success(
                f"""
                Patient: {patient_name}

                Result: Not Melanoma

                Confidence: {confidence:.2f}%
                """
            )

# =========================
# MODEL PERFORMANCE
# =========================
elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "Training Accuracy",
        "73.36%"
    )

    col2.metric(
        "Validation Accuracy",
        "77.48%"
    )

    st.divider()

    st.markdown("""
    ### Model Summary

    - CNN Based Architecture (DenseNet121)
    - Binary Classification
    - Input Size: 224 × 224
    - Optimizer: Adam
    - Loss Function: Binary Crossentropy
    
    ### Prediction Logic
    
    - Prediction > 0.5 → **Melanoma Detected**
    - Prediction ≤ 0.5 → **Not Melanoma**
    """)

# =========================
# DATASET PAGE
# =========================
elif page == "📂 Dataset Info":

    st.title("📂 Dataset Information")

    st.markdown("""
    ### Dataset Structure

    Dataset/
    ├── Melanoma
    └── NotMelanoma

    ### Classes

    • Melanoma
    • Not Melanoma

    ### Total Images

    • 10,682 Images

    ### Source

    HAM10000 inspired skin lesion dataset.
    """)

# =========================
# ABOUT PAGE
# =========================
elif page == "👨‍💻 About":

    st.title("👨‍💻 About Project")

    st.markdown("""
    ## Melanoma Skin Cancer Detection

    This application uses Artificial Intelligence and Deep Learning
    to classify skin lesion images.

    ### Technologies Used

    - Python
    - TensorFlow / Keras
    - Streamlit
    - NumPy
    - Pillow (PIL)

    ### Model Architecture

    - DenseNet121 with Transfer Learning
    - Input: 224 × 224 RGB images
    - Output: Binary classification (Melanoma / Not Melanoma)
    - Training Dataset: HAM10000

    ### Developed By

    Koushik Arnuri

    ### Important Disclaimer

    ⚠️ **MEDICAL DISCLAIMER**

    This application is intended for **educational and research purposes only**.

    It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.

    Always consult a qualified dermatologist or healthcare professional for medical concerns.
    """)
