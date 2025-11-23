# These are the tools we need
from PIL import Image   # To open and resize pictures
import numpy as np      # To turn pictures into numbers
import tensorflow as tf # To talk to your TFLite robot
import streamlit as st  # To make the web app interactive

# Load the robot’s brain
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

# Get info about what the robot expects
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# This function turns any picture into something the robot can understand
IMG_HEIGHT = 224
IMG_WIDTH = 224

def preprocess_image(img_path):
    # Open the picture
    img = Image.open(img_path).convert('RGB')
    
    # Resize to the size the robot expects
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Turn the picture into numbers 0-255
    img_array = np.array(img, dtype=np.uint8)
    
    # Add a "batch" so the robot knows there is 1 picture
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Streamlit part: let user upload a picture
st.title("My AI Picture Robot")
uploaded_file = st.file_uploader("Choose a picture...", type=["jpg","png"])

if uploaded_file is not None:
    # Preprocess the picture for the robot
    img = preprocess_image(uploaded_file)
    
    # Give the picture to the robot
    interpreter.set_tensor(input_details[0]['index'], img)
    
    # Make the robot think
    interpreter.invoke()
    
    # Get the robot's answer
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Show the answer
    st.write("The robot thinks:", output_data)
    
    # Also show the picture
    st.image(uploaded_file, caption="Your picture", use_column_width=True)

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
