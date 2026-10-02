import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Manufacturing Defect Detection",
    page_icon="🔍",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🔍 Manufacturing Defect Detection")

st.write(
    "Upload a steel surface image to predict "
    "the type of manufacturing defect."
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "resnet50_finetuned.keras"
    )

    return model


model = load_model()


# --------------------------------------------------
# Class Names
# --------------------------------------------------

classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]


# --------------------------------------------------
# Image Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a steel defect image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    # Resize
    image_resized = image.resize(
        (224, 224)
    )

    # Convert to NumPy array
    image_array = np.array(
        image_resized
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        probabilities
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = (
        probabilities[predicted_index] * 100
    )


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.success(
        f"Predicted Defect: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # --------------------------------------------------
    # Show All Probabilities
    # --------------------------------------------------

    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(classes):

        probability = probabilities[i] * 100

        st.write(
            f"{class_name}: {probability:.2f}%"
        )

        st.progress(
            float(probabilities[i])
        )