import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Melanoma Skin Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# -----------------------
# CUSTOM CSS
# -----------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#0E76A8;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:gray;
}

.result-box {
    padding:20px;
    border-radius:10px;
    background-color:#ffffff;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_my_model():
    return load_model("melanoma_model.keras")

model = load_my_model()

# -----------------------
# HEADER
# -----------------------
st.markdown(
    '<p class="title">🩺 Melanoma Skin Cancer Detection</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI Powered Skin Lesion Classification System</p>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("📌 Project Information")

st.sidebar.info("""
### Model Details

- Model: CNN
- Classes: Melanoma / Not Melanoma
- Image Size: 128 x 128
- Framework: TensorFlow
- Interface: Streamlit
""")

st.sidebar.success("Developed by Koushik Arnuri")

# -----------------------
# MAIN LAYOUT
# -----------------------
col1, col2 = st.columns([1,1])

with col1:

    st.subheader("📤 Upload Skin Lesion Image")

    uploaded_file = st.file_uploader(
        "Upload JPG, JPEG or PNG Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

with col2:

    st.subheader("🔍 Prediction Result")

    if uploaded_file:

        img = image.resize((128,128))

        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing Skin Lesion..."):

            time.sleep(2)

            prediction = model.predict(img_array)[0][0]

        if prediction < 0.5:

            confidence = (1-prediction)*100

            st.error("⚠️ Melanoma Detected")

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

        else:

            confidence = prediction*100

            st.success("✅ Not Melanoma")

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

# -----------------------
# FOOTER
# -----------------------
st.divider()

st.markdown("""
### 📖 About

Melanoma is a serious form of skin cancer.
This AI system assists in identifying suspicious lesions using Deep Learning techniques.

⚠️ This tool is intended for educational and research purposes only and should not replace professional medical diagnosis.
""")
