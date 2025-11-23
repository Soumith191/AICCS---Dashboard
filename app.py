import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("Solar Panel AI Dashboard — AICCS System")

# Load model
model = tf.keras.models.load_model("keras_model.h5")

# Load labels
with open("labels.txt") as f:
    labels = [l.strip() for l in f.readlines()]

st.header("1. Upload Solar Panel Image")
pic = st.file_uploader("Upload image...", type=["jpg","png","jpeg"])

dust_detected = False

if pic:
    image = Image.open(pic).convert("RGB")
    st.image(image, caption="Your image", use_column_width=True)

    img = image.resize((224,224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr)
    idx = np.argmax(preds)
    label = labels[idx]
    conf = preds[0][idx]

    st.write("Prediction:", label)
    st.write("Confidence:", round(conf*100,2), "%")

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

st.header("Efficiency Facts")
st.write("- Mist cooling: +6–9% efficiency")
st.write("- PCM cooling: +3–5% efficiency")
st.write("- Dust cleaning: restores up to 25% power")
