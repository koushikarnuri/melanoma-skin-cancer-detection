import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Melanoma AI System",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_my_model():
    return load_model("melanoma_model.keras")

model = load_my_model()

labels = ["Melanoma", "NotMelanoma"]

st.sidebar.title("🩺 Melanoma AI")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔍 Predict", "📊 Model Performance", "📂 Dataset Info", "👨‍💻 About"]
)

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

elif page == "🔍 Predict":
    st.title("🩺 Skin Cancer Screening")
    st.markdown("### 👤 Patient Information")
    c1, c2 = st.columns(2)
    with c1:
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=25)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        patient_id = st.text_input("Patient ID", placeholder="Optional")
    
    st.divider()
    uploaded_file = st.file_uploader("📤 Upload Skin Lesion Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.success(f"Hi {patient_name}, thank you for uploading your skin lesion image.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        
        img = np.array(image)
        img = cv2.resize(img, (32, 32))
        img = img.reshape(1, 32, 32, 3)
        img = img.astype('float32')
        img = img / 255.0
        
        raw_predict = model.predict(img, verbose=0)
        predicted_class = np.argmax(raw_predict)
        confidence = np.max(raw_predict) * 100
        
        st.write(f"**Raw Output:** {raw_predict}")
        st.write(f"**Predicted Index:** {predicted_class}")
        st.write(f"**Prediction:** {labels[predicted_class]}")
        
        with col2:
            st.subheader("🧾 Medical Screening Report")
            st.write(f"**Patient Name:** {patient_name}")
            st.write(f"**Patient ID:** {patient_id}")
            st.write(f"**Age:** {patient_age}")
            st.write(f"**Gender:** {gender}")
            st.divider()
            
            if predicted_class == 0:
                st.error("⚠️ MELANOMA DETECTED")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                st.warning("""
                Recommendation:
                • Consult a dermatologist immediately.
                • Perform additional clinical examination.
                • Consider biopsy confirmation.
                """)
            else:
                st.success("✅ NOT MELANOMA")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                st.info("""
                Recommendation:
                • No melanoma detected by the model.
                • Continue regular skin monitoring.
                • Consult a dermatologist if symptoms persist.
                """)
        
        st.divider()
        st.subheader("📋 Final Result")
        
        if predicted_class == 0:
            st.error(f"""
            Patient: {patient_name}
            Result: Melanoma Detected
            Confidence: {confidence:.2f}%
            """)
        else:
            st.success(f"""
            Patient: {patient_name}
            Result: Not Melanoma
            Confidence: {confidence:.2f}%
            """)

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Training Accuracy", "73.36%")
    col2.metric("Validation Accuracy", "77.48%")
    st.divider()
    st.markdown("""
    ### Model Summary
    - DenseNet121 with Transfer Learning
    - 32 × 32 Input
    - AveragePooling2D → Flatten → Dense(128) → Dense(2, softmax)
    """)

elif page == "📂 Dataset Info":
    st.title("📂 Dataset Information")
    st.markdown("""
    ### Classes
    • Melanoma
    • Not Melanoma
    
    ### Total Images
    • 10,682 Images
    
    ### Source
    HAM10000 inspired skin lesion dataset.
    """)

elif page == "👨‍💻 About":
    st.title("👨‍💻 About Project")
    st.markdown("""
    ## Melanoma Skin Cancer Detection
    Developed By: Koushik Arnuri
    
    ### Disclaimer
    Educational & research purposes only. Not for medical diagnosis.
    """)
