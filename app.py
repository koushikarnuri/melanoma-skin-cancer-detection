import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(page_title="Melanoma Detection", layout="wide")
st.title("🩺 Melanoma Detection Using DenseNet121")

categories = ['Melanoma', 'NotMelanoma']

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_dataset():
    path = "Dataset"
    X, Y = [], []
    labels = sorted(os.listdir(path))

    for label in labels:
        folder = os.path.join(path, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            X.append(img)
            Y.append(labels.index(label))

    X = np.array(X, dtype="float32") / 255.0
    Y = to_categorical(np.array(Y))
    return X, Y, labels

# --------------------------------------------------
# Build DenseNet Model
# --------------------------------------------------
def build_model(num_classes):
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(32, 32, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = AveragePooling2D(pool_size=(1, 1))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Menu")
option = st.sidebar.radio("Select Option", ["Train Model", "Predict Image"])

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
if option == "Train Model":
    st.header("📊 Train DenseNet121 Model")

    X, Y, labels = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42
    )

    model = build_model(y_train.shape[1])

    if st.button("Start Training"):
        checkpoint = ModelCheckpoint(
            "model/densenet_weights.hdf5",
            save_best_only=True,
            verbose=1
        )

        with st.spinner("Training in progress..."):
            model.fit(
                X_train,
                y_train,
                epochs=20,
                batch_size=64,
                validation_data=(X_test, y_test),
                callbacks=[checkpoint]
            )

        st.success("✅ Training Completed")

        # Evaluation
        preds = np.argmax(model.predict(X_test), axis=1)
        true = np.argmax(y_test, axis=1)

        acc = accuracy_score(true, preds) * 100
        prec = precision_score(true, preds, average="macro") * 100
        rec = recall_score(true, preds, average="macro") * 100
        f1 = f1_score(true, preds, average="macro") * 100

        st.subheader("📈 Performance Metrics")
        st.write(f"**Accuracy:** {acc:.2f}%")
        st.write(f"**Precision:** {prec:.2f}%")
        st.write(f"**Recall:** {rec:.2f}%")
        st.write(f"**F1 Score:** {f1:.2f}%")

        # Confusion Matrix (PURE MATPLOTLIB)
        cm = confusion_matrix(true, preds)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j],
                        ha="center", va="center", color="black")

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# --------------------------------------------------
# PREDICT IMAGE
# --------------------------------------------------
if option == "Predict Image":
    st.header("🔍 Melanoma Prediction")

    uploaded_file = st.file_uploader(
        "Upload Skin Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = load_img(uploaded_file, target_size=(32, 32))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = build_model(len(categories))
        model.load_weights("model/densenet_weights.hdf5")

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.image(uploaded_file, caption="Uploaded Image", width=250)
        st.success(f"🧠 Prediction: **{categories[class_index]}**")
        st.info(f"Confidence: {confidence:.2f}%")
