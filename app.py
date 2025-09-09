import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5")

model = load_model()

st.title("🧠 MNIST Digit Classifier with TensorFlow + Streamlit")

# Upload image
uploaded_file = st.file_uploader("Upload a digit image (28x28 or larger)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_array = np.array(image.resize((28, 28))) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    if st.button("Predict"):
        prediction = model.predict(img_array)
        st.write(f"Prediction: **{np.argmax(prediction)}**")
