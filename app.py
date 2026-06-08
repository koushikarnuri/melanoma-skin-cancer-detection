import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, AveragePooling2D, Flatten, Dropout
from tensorflow.keras.applications import DenseNet121
from PIL import Image
import numpy as np

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
        st.warning("Could not load weights")
    
    return model

model = load_my_model()

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

        try:
            img = image.resize((32, 32))
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            raw_output = model.predict(img, verbose=0)
            
            melanoma_prob = float(raw_output[0][0])
            not_melanoma_prob = float(raw_output[0][1])

            st.write(f"**Melanoma Prob:** {melanoma_prob:.4f}")
            st.write(f"**Not Melanoma Prob:** {not_melanoma_prob:.4f}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

        with col2:

            st.subheader("🧾 Medical Screening Report")

            st.write(f"**Patient Name:** {patient_name}")
            st.write(f"**Patient ID:** {patient_id}")
            st.write(f"**Age:** {patient_age}")
            st.write(f"**Gender:** {gender}")

            st.divider()

            if melanoma_prob > not_melanoma_prob:

                confidence = melanoma_prob * 100

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

                confidence = not_melanoma_prob * 100

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

        if melanoma_prob > not_melanoma_prob:

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

    - DenseNet121 with Transfer Learning
    - AveragePooling2D → Flatten → Dense(128) → Dense(2)
    - Input Size: 32 × 32
    - 2-Class Softmax Output
    """)

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

    ### Developed By

    Koushik Arnuri

    ### Important Disclaimer

    This application is intended for educational and research purposes only.

    It should NOT be used as a substitute for professional medical advice.
    """)
