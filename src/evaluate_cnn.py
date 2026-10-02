import os

# ---------------------------------------------------------
# SILENCE TENSORFLOW LOGS
# ---------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

tf.get_logger().setLevel(logging.ERROR)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dogs_cats_mobilenetv2.keras"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
IMG_SIZE = (160, 160)
BATCH_SIZE = 16

CLASS_NAMES = [
    "cats",
    "dogs"
]


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ---------------------------------------------------------
# LOAD TEST DATASET
# ---------------------------------------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=False
)


# ---------------------------------------------------------
# GET TRUE LABELS
# ---------------------------------------------------------
y_true = []

for images, labels in test_ds:
    y_true.extend(
        labels.numpy()
    )

y_true = np.array(y_true)


# ---------------------------------------------------------
# GENERATE PREDICTIONS
# ---------------------------------------------------------
predictions = model.predict(
    test_ds,
    verbose=0
)

predicted_probabilities = predictions.ravel()

y_pred = (
    predicted_probabilities >= 0.5
).astype(int)


# ---------------------------------------------------------
# ACCURACY
# ---------------------------------------------------------
accuracy = accuracy_score(
    y_true,
    y_pred
)


# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
)


# ---------------------------------------------------------
# SAVE CLASSIFICATION REPORT
# ---------------------------------------------------------
report_path = os.path.join(
    RESULTS_DIR,
    "cnn_classification_report.txt"
)

with open(
    report_path,
    "w"
) as f:
    f.write(report)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(7, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(
    "Dogs vs Cats - MobileNetV2 Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "cnn_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# FINAL OUTPUT ONLY
# ---------------------------------------------------------
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print()
print("Classification Report:")
print(report)