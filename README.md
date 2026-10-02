# Dogs vs Cats Image Classification using MobileNetV2

A deep learning image classification project that distinguishes between cats and dogs using a MobileNetV2-based convolutional neural network with transfer learning.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-4285F4?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-28A745?style=for-the-badge)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Objective](#project-objective)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Generated Outputs](#generated-outputs)
- [Learning Outcomes](#learning-outcomes)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [Internship Details](#internship-details)

---

## 🎯 Overview

This project is part of the **SkillCraft Technology Machine Learning Internship** and focuses on binary image classification between cats and dogs.

**Workflow:**
- Loading and organizing the dataset
- Splitting data into training and validation sets
- Applying data augmentation
- Using MobileNetV2 pretrained on ImageNet
- Adding a custom binary classification head
- Training the model
- Evaluating performance on test data
- Generating classification report and confusion matrix
- Predicting the class of a single image

---

## 🎯 Project Objective

Build a CNN model capable of classifying whether an image contains a **cat** or a **dog** with high accuracy using transfer learning and image preprocessing.

---

## 🛠 Technologies Used

- Python 3
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Transfer Learning

---

## 🏗 Model Architecture

The model uses MobileNetV2 as the feature extractor, with the pretrained base model frozen and a new binary classification layer added on top.

```text
Input Image (160 x 160 x 3)
         |
         v
Data Augmentation
         |
         v
MobileNetV2 (ImageNet pretrained)
         |
         v
Global Average Pooling
         |
         v
Dropout (0.3)
         |
         v
Dense Layer (1 unit, Sigmoid)
         |
         v
CAT / DOG Prediction
```

---

## 📊 Dataset

The project uses the standard cats-vs-dogs dataset organized as follows:

```text
dataset/
├── train/
│   ├── cats/
│   └── dogs/
├── test/
│   ├── cats/
│   └── dogs/
└── README.md
```

**Dataset Summary:**

| Split | Cats | Dogs | Total |
|-------|-----:|-----:|------:|
| Training | 5,001 | 5,001 | 10,002 |
| Validation | 750* | 750* | 1,500 |
| Testing | 450 | 501 | 951 |

*Validation is created automatically from 15% of the training set during training.

---

## 📁 Project Structure

```text
SCT_ML_Task3/
├── README.md
├── requirements.txt
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
├── models/
│   ├── dogs_cats_mobilenetv2.keras
│   └── dogs_cats_mobilenetv2_final.keras
├── results/
│   ├── cnn_classification_report.txt
│   ├── cnn_confusion_matrix.png
│   ├── cnn_training_accuracy.png
│   └── cnn_training_loss.png
└── src/
    ├── train_cnn.py
    ├── evaluate_cnn.py
    └── predict_cnn.py
```

---

## 📦 Requirements

Make sure Python 3.10+ is installed, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/VigneshVarsha/SCT_ML-3.git
cd SCT_ML-3
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train the model

```bash
python src/train_cnn.py
```

This script:
- Loads the dataset
- Creates the validation split
- Applies augmentation
- Builds the MobileNetV2 model
- Compiles the model
- Trains for 5 epochs
- Saves the trained model in `models/`

### Evaluate the model

```bash
python src/evaluate_cnn.py
```

This generates:
- `results/cnn_confusion_matrix.png`
- `results/cnn_classification_report.txt`

### Predict a single image

```bash
python src/predict_cnn.py "dataset/test/cats/cat_test_00010.jpg"
```

**Example output:**
```text
Prediction: CAT
Confidence: 84.93%
```

Another example:
```bash
python src/predict_cnn.py "dataset/test/dogs/dog_test_00019.jpg"
```

---

## 📈 Model Performance

The trained model achieved strong results on the test set.

| Metric | Value |
|--------|------:|
| **Test Accuracy** | **98.00%** |
| **Test Loss** | **0.0674** |

### Classification Report

```text
              precision    recall  f1-score   support

      cats       0.98      0.98      0.98       450
      dogs       0.98      0.98      0.98       501

      accuracy                           0.98       951
     macro avg       0.98      0.98      0.98       951
  weighted avg       0.98      0.98      0.98       951
```

### Confusion Matrix

```text
                  Predicted
                  Cat    Dog
Actual Cat       440     10
Actual Dog         9    492
```

**Interpretation:**
- ✅ 440 cats correctly classified as cats
- ✅ 492 dogs correctly classified as dogs
- 📊 **932 correct predictions out of 951 total images**

---

## 📂 Generated Outputs

The project creates visual and textual artifacts in the `results/` folder:

| File | Description |
|------|-------------|
| `cnn_training_accuracy.png` | Training accuracy visualization |
| `cnn_training_loss.png` | Training loss visualization |
| `cnn_confusion_matrix.png` | Confusion matrix heatmap |
| `cnn_classification_report.txt` | Detailed classification metrics |

---

## 🎓 Learning Outcomes

This project demonstrates:
- Image classification with deep learning
- CNN architecture design
- Transfer learning using MobileNetV2
- Data augmentation for image data
- Binary classification with sigmoid output
- Model evaluation using precision, recall, F1-score, and accuracy
- Prediction on unseen images
- Saving and loading TensorFlow/Keras models

---

## 🚀 Future Improvements

- Fine-tuning deeper MobileNetV2 layers
- Trying additional pretrained architectures (EfficientNet, ResNet, etc.)
- Increasing image resolution
- Experimenting with more augmentation techniques
- Building a web or GUI-based prediction interface
- Deploying the model as an application (Flask/FastAPI/Streamlit)

---

## 👨‍💻 Author

**Devarakonda Vignesh Varsha**

---

## 📌 Internship Details

| Detail | Information |
|--------|-------------|
| **Internship** | SkillCraft Technology |
| **Domain** | Machine Learning |
| **Task** | Task 03 - Dogs vs Cats Image Classification |
| **Final Test Accuracy** | **98.00%** |

---

*Built with ❤️ during SkillCraft Technology ML Internship*