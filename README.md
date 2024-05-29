# app.pyimport streamlit as st
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Function to predict the label of an image
def predict_image(img, model, label_encoder):
    if img is not None:
        img = cv2.resize(img, (128, 128))
        img_flattened = img.flatten().reshape(1, -1)
        prediction = model.predict(img_flattened)
        label = label_encoder.inverse_transform(prediction)
        return label[0]
    else:
        return "Invalid image"

# Function to display image with prediction
def display_prediction(image, pred_label):
    st.image(image, channels="RGB")
    st.write(f"Predicted: {pred_label}")

# Streamlit app
st.title("Mango Leaf Disease Prediction")

uploaded_files = st.file_uploader("Upload Mango Leaf Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict and display
        pred_label = predict_image(image, model, label_encoder)
        display_prediction(image_rgb, pred_label)
!streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py
