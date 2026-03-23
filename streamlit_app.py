import streamlit as st
import numpy as np
import cv2
import pickle
from PIL import Image
from skimage.feature import hog
from tensorflow.keras.models import load_model
import joblib

# ---------------- LOAD MODELS ----------------

stage1_model = load_model("leg_model_final")

knn_model = joblib.load("knn_pipeline_new.pkl")
svm_model = joblib.load("svm_model_new.pkl")
svm_scaler = joblib.load("svm_scaler_new.pkl")
svm_pca = joblib.load("svm_pca_new.pkl")

# ---------------- PREPROCESS ----------------

def preprocess_stage1(img):
    img = img.resize((160, 160))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_svm(img):
    img = cv2.resize(img, (128, 128))
    img = cv2.fastNlMeansDenoising(img, h=10)
    img = cv2.equalizeHist(img)

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features

# ---------------- LABEL FORMAT ----------------

def format_label(label):
    if isinstance(label, str):
        return label.replace("Grade", "Level")
    return f"Level {label}"

# ---------------- UI ----------------

st.title("Leg Disease Severity Detection System")

if "history" not in st.session_state:
    st.session_state.history = []

name = st.text_input("Enter Name")
age = st.text_input("Enter Age")
gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ---------------- BUTTONS ----------------

col1, col2 = st.columns(2)

predict_btn = col1.button("Predict")
reset_btn = col2.button("Reset")

# ---------------- RESET FUNCTION ----------------

if reset_btn:
    st.session_state.clear()
    st.rerun()

# ---------------- PREDICT ----------------

if predict_btn:

    if uploaded_file is not None:

        # LOADING SPINNER
        with st.spinner("⏳ Predicting... Please wait..."):

            img = Image.open(uploaded_file)

            st.image(img, width="stretch")

            # -------- STAGE 1 --------
            stage1_input = preprocess_stage1(img)
            stage1_result = stage1_model.predict(stage1_input)[0][0]

            if stage1_result > 0.5:
                st.error("The image is not valid for prediction")
            else:
                st.success("Valid Leg Image")

                # -------- KNN --------
                img_knn = img.convert("L").resize((64, 64))
                img_knn_array = np.array(img_knn).flatten().reshape(1, -1)
                knn_pred = knn_model.predict(img_knn_array)[0]

                # -------- SVM --------
                img_gray = np.array(img.convert("L"))
                features = preprocess_svm(img_gray).reshape(1, -1)
                features = svm_scaler.transform(features)
                features = svm_pca.transform(features)
                svm_pred = svm_model.predict(features)[0]

                svm_pred = format_label(svm_pred)

                # -------- OUTPUT --------
                st.subheader("Prediction Result")

                st.write(f"Name: {name}")
                st.write(f"Age: {age}")
                st.write(f"Gender: {gender}")

                st.success(f"KNN Prediction: Level {knn_pred}")
                svm_pred = str(svm_pred).replace("Grade", "Level ")
                st.success(f"SVM Prediction: {svm_pred}")

                st.session_state.history.append({
                     "name": name,
                     "age": age,
                     "gender": gender,
                     "knn": f"Level {knn_pred}",
                     "svm": svm_pred
                })

                st.subheader("Prediction History")

                if len(st.session_state.history) == 0:
                    st.write("No predictions yet")
                else:
                     for i, item in enumerate(st.session_state.history[::-1]):
                         st.write(f"""
                         **{item['name']}** | Age: {item['age']} | Gender: {item['gender']}  
                         KNN: {item['knn']} | SVM: {item['svm']}
                         """)

    else:
        st.warning("⚠ Please upload an image")
