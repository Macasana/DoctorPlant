"""
🌿 PLANTDOCTOR - Streamlit App
Plant Disease Detection using EfficientNetB4
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Import class names from class_names.py
try:
    from class_names import CLASS_NAMES
except ImportError:
    st.error("❌ class_names.py not found!")
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PlantDoctor",
    page_icon="🌿",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌿 PlantDoctor")
st.subheader("AI Plant Disease Detection")

st.write(
    "Upload a plant leaf image or take a photo "
    "to let the AI predict its condition."
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "plant_disease_recog_model_pwp.keras"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Check model file
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model not found:\n\n`{MODEL_PATH}`")
            return None

        # Load Keras model
        model = tf.keras.models.load_model(MODEL_PATH)
        return model

    except Exception as e:
        st.error(f"❌ Error loading model:\n\n{e}")
        return None


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image, model):
    """
    Prepare an image for EfficientNetB4.

    Training image size:
        160 x 160

    Input:
        160 x 160 x 3
    """

    # Get model input shape
    input_shape = model.input_shape
    height = input_shape[1]
    width = input_shape[2]

    # Resize image
    image = image.resize((width, height))

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# PREDICT
# ============================================================

def predict(model, image_array):
    """
    Make prediction using the trained model.
    """
    # Run model
    prediction = model.predict(image_array, verbose=0)[0]

    # Find highest probability
    class_id = int(np.argmax(prediction))

    # Get confidence
    confidence = float(prediction[class_id])

    # Get class name from imported CLASS_NAMES
    if class_id < len(CLASS_NAMES):
        class_name = CLASS_NAMES[class_id]
    else:
        class_name = f"Class {class_id}"

    return class_name, confidence, prediction


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_prediction(model, image):
    """Display prediction results"""

    # Preprocess
    image_array = preprocess_image(image, model)

    # Predict
    with st.spinner("🔄 PlantDoctor is analyzing the leaf..."):
        class_name, confidence, prediction = predict(model, image_array)

    # Results
    st.divider()
    st.subheader("🔍 Diagnosis")

    # Determine healthy/diseased
    is_healthy = "healthy" in class_name.lower()

    if is_healthy:
        st.success(f"🌿 **Healthy Plant**\n\nPrediction: **{class_name}**")
    else:
        st.warning(f"⚠️ **Possible Disease Detected**\n\nPrediction: **{class_name}**")

    # Confidence
    st.metric("🎯 Confidence", f"{confidence * 100:.2f}%")
    st.progress(min(max(confidence, 0.0), 1.0))

    # Top predictions
    st.divider()
    st.subheader("📊 Top Predictions")

    sorted_indices = np.argsort(prediction)[::-1]
    top_n = min(5, len(prediction))

    for position in range(top_n):
        index = int(sorted_indices[position])
        probability = float(prediction[index])
        percentage = probability * 100

        if index < len(CLASS_NAMES):
            name = CLASS_NAMES[index]
        else:
            name = f"Class {index}"

        st.write(f"**{name}** — {percentage:.2f}%")
        st.progress(min(max(probability, 0.0), 1.0))


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    # Load model
    model = load_model()

    # Stop if model failed
    if model is None:
        st.warning("⚠️ PlantDoctor could not start.")
        st.info("""
            Make sure these files are in the same folder as `app.py`:
            • plant_disease_recog_model_pwp.keras
            • class_names.py
        """)
        return

    # Check number of classes
    number_of_classes = len(CLASS_NAMES)

    # Sidebar
    with st.sidebar:
        st.header("🌿 PlantDoctor")
        st.success("Model loaded!")
        st.write(f"**Classes:** {number_of_classes}")
        st.write(f"**Input:** {model.input_shape}")
        st.write("**Model:** EfficientNetB4")
        st.divider()
        st.caption("PlantDoctor uses artificial intelligence to classify plant leaf images.")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📸 Camera", "📁 Upload Image", "ℹ️ About"])

    # ========================================================
    # CAMERA
    # ========================================================
    with tab1:
        st.subheader("📸 Take a Photo")
        st.write("Take a clear picture of the plant leaf.")
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            image = Image.open(camera_image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            st.image(image, caption="Captured Leaf", use_container_width=True)
            display_prediction(model, image)

    # ========================================================
    # UPLOAD IMAGE
    # ========================================================
    with tab2:
        st.subheader("📁 Upload an Image")
        st.write("Upload a JPG, JPEG or PNG image of a plant leaf.")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode != "RGB":
                image = image.convert("RGB")
            st.image(image, caption="Uploaded Leaf", use_container_width=True)
            display_prediction(model, image)

    # ========================================================
    # ABOUT
    # ========================================================
    with tab3:
        st.subheader("ℹ️ About PlantDoctor")

        st.markdown("""
            ## 🌿 PlantDoctor

            PlantDoctor is an AI-powered plant disease classification application.

            The model was trained using plant leaf images and uses **EfficientNetB4** with transfer learning.

            ### 🤖 Model Information
            - Architecture: **EfficientNetB4**
            - Input size: **160 × 160**
            - Number of classes: **38**
            - Framework: **TensorFlow / Keras**
            - Model format: **.keras**
            - Pre-trained weights: **ImageNet**

            ### 📷 How to use
            1. Take a photo of a plant leaf, or upload an existing image.
            2. PlantDoctor resizes the image to the model's required input size.
            3. The AI analyzes the image.
            4. The predicted class is displayed.
            5. The confidence score and top predictions are shown.

            ### ⚠️ Important
            PlantDoctor provides an AI-based prediction. It should not be treated as a definitive professional agricultural diagnosis.
        """)

        st.divider()
        st.subheader("🌱 Supported Classes")

        # Show classes from CLASS_NAMES
        for number, class_name in enumerate(CLASS_NAMES, start=1):
            st.write(f"{number}. {class_name}")


# ============================================================
# START APP
# ============================================================

if __name__ == "__main__":
    main()