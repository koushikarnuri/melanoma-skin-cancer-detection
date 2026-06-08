import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, Flatten, Dropout
from tensorflow.keras.applications import DenseNet121
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
    densenet = DenseNet121(
        input_shape=(32, 32, 3),
        include_top=False,
        weights='imagenet'
    )
    
    for layer in densenet.layers:
        layer.trainable = False
    
    x = densenet.output
    x = AveragePooling2D(pool_size=(1, 1))(x)
    x = Flatten(name="flatten")(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=densenet.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    try:
        model.load_weights("melanoma_model.keras")
    except:
        st.warning("⚠️ Model weights not found or incompatible")
    
    return model

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
        
        try:
            img = np.array(image)
            img = cv2.resize(img, (32, 32))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            raw_predict = model.predict(img, verbose=0)
            predicted_class = np.argmax(raw_predict)
            confidence = np.max(raw_predict) * 100
            
            st.write(f"**Raw:** {raw_predict}")
            st.write(f"**Class:** {labels[predicted_class]}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()
        
        with col2:
            st.subheader("🧾 Report")
            st.write(f"**Name:** {patient_name}")
            st.write(f"**ID:** {patient_id}")
            st.write(f"**Age:** {patient_age}")
            st.divider()
            
            if predicted_class == 0:
                st.error("⚠️ MELANOMA DETECTED")
                st.metric("Confidence", f"{confidence:.2f}%")
                st.warning("Consult a dermatologist immediately.")
            else:
                st.success("✅ NOT MELANOMA")
                st.metric("Confidence", f"{confidence:.2f}%")
                st.info("No melanoma detected. Continue regular monitoring.")

elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "77.48%")
    col2.metric("Framework", "DenseNet121")

elif page == "📂 Dataset Info":
    st.title("📂 Dataset")
    st.write("Melanoma: Class 0")
    st.write("Not Melanoma: Class 1")
    st.write("Total: 10,682 images")

elif page == "👨‍💻 About":
    st.title("About")
    st.write("Melanoma Detection using Deep Learning")
    st.write("Developer: Koushik Arnuri")
    st.write("⚠️ Educational use only - NOT for medical diagnosis")
