import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit import divider
import gdown

working_dir = os.path.dirname(os.path.abspath(__file__))

# model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
# model = tf.keras.models.load_model(model_path)


# https://drive.google.com/file/d/1FVVhqamVYyfNI3NBi7grVgy0sxMNFdg3/view?usp=sharing

def download_model():
    # Google Drive file ID
    file_id = "1FVVhqamVYyfNI3NBi7grVgy0sxMNFdg3"
    # The direct download link format for Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    # Output file path
    output = "plant_disease_prediction_model.h5"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Call this function to download the model at runtime
download_model()


# Load the model
model = tf.keras.models.load_model("plant_disease_prediction_model.h5")


# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Apple Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        # 150, 150 is for display the image in frontend
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            apple = prediction[:5] == "Apple"
            if (apple):
                st.success(f'Prediction: {str(prediction)}')
            else:
                st.error(f'It is not an Apple Leaf')
            
st.subheader("", divider= "red")
st.subheader("Application Development (Deep Learning) IV-I ")
st.markdown("Midhun Naga Sai. M -2111CS010283")
st.markdown("Nagendra Kumar. J -2111CS010302")
st.markdown("Debasish Nayak -2111CS010312")
st.markdown("Partha Saradhi Reddy. C -2111CS010348")
st.markdown("Nikhila. G -2111CS010323")

