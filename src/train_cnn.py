import os

# ---------------------------------------------------------
# SILENCE TENSORFLOW LOGS
# ---------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging
import tensorflow as tf
import matplotlib.pyplot as plt

tf.get_logger().setLevel(logging.ERROR)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "train"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
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
EPOCHS = 5
SEED = 42


# ---------------------------------------------------------
# LOAD TRAINING DATA
# ---------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="training",
    seed=SEED,
    shuffle=True,
    verbose=False
)


# ---------------------------------------------------------
# LOAD VALIDATION DATA
# ---------------------------------------------------------
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=SEED,
    shuffle=True,
    verbose=False
)


# ---------------------------------------------------------
# LOAD TEST DATA
# ---------------------------------------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=False
)


# ---------------------------------------------------------
# DATA AUGMENTATION
# ---------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.1
    ),

    tf.keras.layers.RandomContrast(
        0.1
    )
])


# ---------------------------------------------------------
# MOBILENETV2 PREPROCESSING
# ---------------------------------------------------------
preprocess_input = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)


# ---------------------------------------------------------
# BASE MODEL
# ---------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),

    include_top=False,

    weights="imagenet"
)

base_model.trainable = False


# ---------------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------------
inputs = tf.keras.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    )
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.3
)(x)

outputs = tf.keras.layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# ---------------------------------------------------------
# COMPILE
# ---------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ---------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------
checkpoint_path = os.path.join(
    MODEL_DIR,
    "dogs_cats_mobilenetv2.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=0
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        mode="max",
        verbose=0
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=0
    )
]


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------
history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=2
)


# ---------------------------------------------------------
# SAVE FINAL MODEL
# ---------------------------------------------------------
final_model_path = os.path.join(
    MODEL_DIR,
    "dogs_cats_mobilenetv2_final.keras"
)

model.save(
    final_model_path
)


# ---------------------------------------------------------
# TEST EVALUATION
# ---------------------------------------------------------
test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=0
)


# ---------------------------------------------------------
# TRAINING ACCURACY GRAPH
# ---------------------------------------------------------
plt.figure(
    figsize=(8, 6)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "MobileNetV2 Training and Validation Accuracy"
)

plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    RESULTS_DIR,
    "cnn_training_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# TRAINING LOSS GRAPH
# ---------------------------------------------------------
plt.figure(
    figsize=(8, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "MobileNetV2 Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    RESULTS_DIR,
    "cnn_training_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# FINAL OUTPUT ONLY
# ---------------------------------------------------------
print()
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")