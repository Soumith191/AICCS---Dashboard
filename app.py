import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("AICCS Solar Panel AI — TFLite Quantized Version")

# ----------------------------
# Load TFLite quantized model
# ----------------------------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels (dusty vs clean)
with open("labels.txt") as f:
    labels = [l.strip() for l in f.readlines()]

# ----------------------------
# Helper: preprocess for quantized model
# ----------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224

def preprocess_image_quantized(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.uint8)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# 1️⃣ Upload Image
# ----------------------------
st.header("1. Upload Solar Panel Image")
pic = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

dust_detected = False

if pic:
    # Show uploaded image
    image = Image.open(pic).convert("RGB")
    st.image(image, use_column_width=True)

    # Preprocess image
    img = preprocess_image_quantized(pic)

    # Feed to model
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = np.argmax(preds)
    label = labels[idx]
    conf = preds[idx]

    st.write("Prediction:", label)
    st.write("Confidence:", round(float(conf)*100, 2), "%")

    if "dust" in label.lower():
        dust_detected = True

# ----------------------------
# 2️⃣ Temperature Input
# ----------------------------
st.header("2. Temperature Input")
temperature = st.slider("Select temperature (°C)", 20, 70, 35)

# ----------------------------
# 3️⃣ AI Decision with Color & Emoji
# ----------------------------
st.header("3. AI Decision")

if dust_detected:
    st.markdown(
        "<div style='padding:20px; background-color:#FF6B6B; color:white; font-size:24px; border-radius:10px; text-align:center;'>🧹 Action: Electrostatic Cleaning Needed!</div>",
        unsafe_allow_html=True
    )
elif temperature > 50:
    st.markdown(
        "<div style='padding:20px; background-color:#FFD93D; color:black; font-size:24px; border-radius:10px; text-align:center;'>💧 Action: Mist Cooling Recommended!</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div style='padding:20px; background-color:#6BCB77; color:white; font-size:24px; border-radius:10px; text-align:center;'>✅ Action: No Action Needed!</div>",
        unsafe_allow_html=True
    )
