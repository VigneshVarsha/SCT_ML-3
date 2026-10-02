import os

# ---------------------------------------------------------
# SILENCE TENSORFLOW LOGS
# ---------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import sys
import logging
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dogs_cats_mobilenetv2.keras"
)

IMG_SIZE = (160, 160)


# ---------------------------------------------------------
# CHECK IMAGE PATH
# ---------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python src\\predict_cnn.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.isfile(image_path):
    print("Error: Image not found.")
    sys.exit(1)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------
try:
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

except Exception:
    print("Error: Could not load image.")
    sys.exit(1)


# ---------------------------------------------------------
# PREPROCESS
# ---------------------------------------------------------
image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
prediction = model.predict(
    image_array,
    verbose=0
)[0][0]


if prediction >= 0.5:
    predicted_class = "DOG"
    confidence = prediction * 100
else:
    predicted_class = "CAT"
    confidence = (1 - prediction) * 100


# ---------------------------------------------------------
# FINAL OUTPUT ONLY
# ---------------------------------------------------------
print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")