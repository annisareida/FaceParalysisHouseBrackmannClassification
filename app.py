import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('final_modelYFPMEIIBINER100modelfit2.h5', compile=False)

model = load_model()
class_names = ['Normal', 'Paralysis']

# Prediksi
def predict_image(img_pil):
    img = img_pil.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = int(prediction[0][0] >= 0.5)
    class_name = class_names[predicted_class]

    return class_name, prediction[0][0]

# UI Streamlit
st.title("Face Paralysis Detection")
st.write("Upload an image or use your webcam to classify face as **Normal** or **Paralysis**.")

# Pilih sumber gambar
input_type = st.radio("Select input method:", ['Upload Image', 'Use Webcam'])

image_source = None
if input_type == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_source = Image.open(uploaded_file).convert('RGB')
elif input_type == 'Use Webcam':
    webcam_image = st.camera_input("Take a photo")
    if webcam_image:
        image_source = Image.open(webcam_image).convert('RGB')

# Prediksi dan hasil
if image_source:
    st.image(image_source, caption="Input Image", use_container_width=True)


    if st.button("Predict"):
        label, confidence = predict_image(image_source)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence:.2f}**")
