# 🩺 Melanoma Detection Using DenseNet121

## 📌 Overview

This project is a Deep Learning-based web application developed to detect **Melanoma Skin Cancer** from dermoscopic skin lesion images using the **DenseNet121** architecture.

The application allows users to:

* Train a Deep Learning model
* Upload skin lesion images
* Predict whether the lesion is:

  * **Melanoma**
  * **Not Melanoma**

The project is implemented using **TensorFlow, Keras, OpenCV, and Streamlit**.

---

# 🚀 Features

* 📂 Upload skin lesion images
* 🧠 DenseNet121 transfer learning model
* 📊 Model training and evaluation
* 📈 Accuracy, Precision, Recall, and F1-Score metrics
* 🔍 Real-time melanoma prediction
* 📉 Confusion matrix visualization
* 🖥️ Interactive Streamlit web interface

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit
* Jupyter Notebook

---

# 📂 Project Structure

```text
Melanoma/
│
├── Dataset/
│   ├── Melanoma/
│   └── NotMelanoma/
│
├── model/
│   └── densenet_model.keras
│
├── testImages/
├── main.py
├── requirements.txt
├── MelanomaDetection.ipynb
├── README.md
└── screenshots/
```

---

# 📊 Dataset

This project uses the Melanoma dataset available on Kaggle:

Dataset Link:
https://www.kaggle.com/datasets/acsmanikoushik/melanoma-dataset
https://www.kaggle.com/datasets/drscarlat/melanoma

The original data is based on the **HAM10000 (Human Against Machine with 10,000 Training Images)** dataset.

The dataset contains dermoscopic images of skin lesions curated and normalized for:

* Luminosity
* Color consistency
* Resolution quality

More than 50% of the diagnoses were validated using histopathology, while the remaining cases were verified by expert dermatologists.

---

## Dataset Simplification

Instead of classifying seven different skin lesion categories, this project simplifies the classification into:

* Melanoma
* Not Melanoma

### Original Dataset Distribution

* 1,113 Melanoma images
* 8,902 Not Melanoma images

### Data Augmentation

Data augmentation techniques were applied to balance the dataset.

After augmentation:

* 8,903 Melanoma images
* 8,902 Not Melanoma images

Final dataset size:

* Approximately **17,805 images**

⚠️ Dataset is not uploaded to GitHub because it contains more than 10,000 images and exceeds GitHub storage recommendations.

---

# 🧠 Model Architecture

The project uses **DenseNet121** with transfer learning.

### Model Workflow

1. Load Dataset
2. Image Preprocessing
3. Train-Test Split
4. DenseNet121 Feature Extraction
5. CNN Classification Layers
6. Model Training
7. Performance Evaluation
8. Image Prediction

---

# 📊 Overall Model Performance

The DenseNet121 model was trained and evaluated on the balanced melanoma dataset containing approximately **17,805 dermoscopic skin lesion images**.

## Total Dataset Distribution

| Category         | Total Images |
| ---------------- | ------------ |
| Melanoma         | 8,903        |
| Not Melanoma     | 8,902        |
| **Total Images** | **17,805**   |

---

## 📈 Performance Metrics

| Metric    | Performance |
| --------- | ----------- |
| Accuracy  | 96%         |
| Precision | 95%         |
| Recall    | 96%         |
| F1-Score  | 95%         |

---

## 📉 Evaluation Details

The performance metrics were calculated using:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

A confusion matrix was generated to analyze prediction accuracy between:

* Melanoma images
* Not Melanoma images

The model demonstrated strong capability in distinguishing melanoma skin lesions from non-melanoma lesions using Deep Learning and transfer learning with DenseNet121.

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Melanoma.git
cd Melanoma
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run main.py
```

---

# 🖥️ Streamlit Application

The Streamlit application provides two main options:

## 1️⃣ Train Model

* Loads dataset
* Trains DenseNet121 model
* Saves best model
* Displays evaluation metrics

## 2️⃣ Predict Image

* Upload skin lesion image
* Predict melanoma status
* Display confidence score

---

# 📸 Screenshots

Add screenshots of:

* Training page
* Prediction page
* Confusion matrix
* Output predictions

inside the `screenshots/` folder.

---

# 🔮 Future Improvements

* Improve model accuracy
* Add multiple skin disease detection
* Deploy on cloud platforms
* Mobile application integration
* Real-time webcam prediction
* Explainable AI integration

---

# 👨‍💻 Author

**Koushik Arnuri**

---

# 📜 License

This project is developed for educational and research purposes only.
