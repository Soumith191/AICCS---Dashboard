import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("AICCS Solar Panel AI — TFLite Version")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt") as f:
    labels = [l.strip() for l in f.readlines()]

st.header("1. Upload Solar Panel Image")
pic = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

dust_detected = False

if pic:
    image = Image.open(pic).convert("RGB")
    st.image(image, use_column_width=True)

    img = image.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]

    idx = np.argmax(preds)
    label = labels[idx]
    conf = preds[idx]

    st.write("Prediction:", label)
    st.write("Confidence:", round(float(conf)*100, 2), "%")

    if "dust" in label.lower():
        dust_detected = True

st.header("2. Temperature Input")
temperature = st.slider("Select temperature (°C)", 20, 70, 35)

st.header("3. AI Decision")
if dust_detected:
    st.error("Action: Electrostatic Cleaning")
elif temperature > 50:
    st.warning("Action: Mist Cooling")
else:
    st.success("Action: No Action Needed")
