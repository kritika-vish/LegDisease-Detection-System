from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import pickle
from PIL import Image
from skimage.feature import hog
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ================= LOAD MODELS =================

# Stage 1 model (Leg / Not Leg)
stage1_model = load_model("leg_model.h5")

# Stage 2 models
knn_model = pickle.load(open("knn_pipeline.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))
svm_scaler = pickle.load(open("svm_scaler.pkl", "rb"))
svm_pca = pickle.load(open("svm_pca.pkl", "rb"))

# ================= PREPROCESS FUNCTIONS =================

# Stage 1 preprocessing
def preprocess_stage1(file):

    img = Image.open(file).convert("RGB")
    img = img.resize((160,160))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# Stage 2 preprocessing (SVM)
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


# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    # ================= STAGE 1 =================
    file.seek(0)
    img_stage1 = preprocess_stage1(file)

    stage1_pred = stage1_model.predict(img_stage1)[0][0]

    # If image is NOT leg
    if stage1_pred > 0.5:
         return jsonify({
            "message": "The image is not valid for prediction"
         })


    # ================= STAGE 2 =================

    # -------- KNN PROCESS --------
    file.seek(0)
    img_pil = Image.open(file).convert("L")
    img_knn = img_pil.resize((64, 64))
    img_knn_array = np.array(img_knn).flatten().reshape(1, -1)

    knn_pred = knn_model.predict(img_knn_array)[0]

    # -------- SVM PROCESS --------
    file.seek(0)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_svm = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    features = preprocess_svm(img_svm).reshape(1, -1)
    features = svm_scaler.transform(features)
    features = svm_pca.transform(features)

    svm_pred = svm_model.predict(features)[0]

    # Grade2 → Level 2 convert
    svm_pred = str(svm_pred).replace("Grade", "Level ")

    return jsonify({
        "KNN Prediction": f"Level {knn_pred}",
        "SVM Prediction": svm_pred
    })


if __name__ == "__main__":
    app.run(debug=True)