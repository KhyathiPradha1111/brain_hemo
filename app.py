import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load your saved model
model = load_model('model\hemo_model (1).h5')

# Set page config
st.set_page_config(page_title="Brain Hemorrhage Detection", layout="centered")

# Sidebar for navigation
page = st.sidebar.selectbox("Select a page:", ["Home", "Prediction"])

# Define hemorrhage classes
hemorrhage_classes = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# Define Grad-CAM function
def predict_and_gradcam(image, model, last_conv_layer_name="conv5_block3_out"):
    img_input = np.expand_dims(image, axis=0)

    predictions = model.predict(img_input)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = hemorrhage_classes[predicted_class_index]

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        class_channel = predictions[:, predicted_class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        superimposed_img = heatmap * alpha + np.uint8(img * 255)
        return superimposed_img

    overlay_img = overlay_heatmap(image, heatmap)

    return overlay_img, predicted_class_name

# Home Page
if page == "Home":
    st.title("🧠 Brain Hemorrhage Detection App")
    st.subheader("Welcome!")
    st.write("""
    This app detects different types of brain hemorrhages from CT scan images using Deep Learning and explains the prediction using Grad-CAM visualization.

    💻 **Upload** a CT scan  
    🧠 **Predict** hemorrhage type  
    🔥 **Visualize** model focus with Grad-CAM heatmap
    """)
    st.write("---")
    st.info("➡️ Go to the 'Prediction' page from the sidebar to try it out!")

# Prediction Page
elif page == "Prediction":
    st.title("🧪 Predict Hemorrhage Type and Generate Grad-CAM")

    uploaded_file = st.file_uploader("Upload a brain CT scan (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0

        st.image(uploaded_file, caption="Uploaded CT Scan", use_column_width=True)

        with st.spinner('Predicting and generating Grad-CAM...'):
            overlay_img, predicted_class_name = predict_and_gradcam(image, model)

        st.success(f"Prediction: {predicted_class_name}")
        st.image(overlay_img.astype('uint8'), caption='Grad-CAM Heatmap', use_column_width=True)
