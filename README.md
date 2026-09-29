# 🤟 Sign Language Detection using MobileNet

> An image classification project for recognizing **24 SIBI (Sistem Isyarat Bahasa Indonesia) hand-sign classes** using **MobileNet** and TensorFlow.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-red?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Detection-green)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-98%25-success)

## 📌 Overview

This project builds an image classification model to recognize Indonesian Sign Language (SIBI) hand signs representing **24 alphabet classes**.

The workflow combines **MediaPipe Hands** for hand localization and **MobileNet** as the image classification backbone. The images are preprocessed, augmented, and then used to train a neural network for multi-class classification.

The model achieved approximately **98% validation accuracy** on 1,056 validation images.

---

## ✨ Highlights

- 🤟 **24 SIBI alphabet classes**
- 🖼️ **5,280 images** processed from the dataset
- ✋ MediaPipe-based hand detection and cropping
- 🎨 Background augmentation
- 🔄 Image augmentation during training
- 🧠 MobileNet pretrained with ImageNet weights
- 📊 Classification Report & Confusion Matrix
- 📈 ROC Curve and AUC evaluation
- 🎯 **98% validation accuracy**
- ⭐ **0.9999 average macro AUC**

---

## 🧩 Classes

The model recognizes the following 24 classes:

```text
A  B  C  D  E  F  G  H  I  K  L  M
N  O  P  Q  R  S  T  U  V  W  X  Y
```

> Note: The dataset used in this notebook contains 24 classes.

---

## 🔄 Project Workflow

```text
📁 SIBI Dataset
       │
       ▼
🖼️ Load Images
       │
       ▼
✋ MediaPipe Hand Detection
       │
       ├── Hand detected → Crop hand region
       │
       └── Not detected → Fallback center crop
       │
       ▼
🎨 Background Augmentation
       │
       ▼
🌫️ Grayscale + Gaussian Blur
       │
       ▼
📐 Resize to 224 × 224
       │
       ▼
⚙️ MobileNet Preprocessing
       │
       ▼
✂️ Train / Validation Split
       │
       ▼
🔄 Image Augmentation
       │
       ▼
🧠 MobileNet Model
       │
       ▼
🏋️ Model Training
       │
       ▼
📊 Evaluation
       │
       ├── Classification Report
       ├── Confusion Matrix
       └── ROC & AUC
```

---

## 🧠 Model Architecture

The project uses **MobileNet** with ImageNet pretrained weights as the base model.

The classification head consists of:

- `GlobalMaxPooling2D`
- `BatchNormalization`
- `Dense(1024, ReLU)`
- `Dropout(0.3)`
- `BatchNormalization`
- `Dense(512, ReLU)`
- `Dense(24, Softmax)`

The last 80 layers of the MobileNet base model remain trainable for fine-tuning.

### Training configuration

| Parameter | Value |
|---|---|
| Input size | `224 × 224 × 3` |
| Batch size | `32` |
| Maximum epochs | `50` |
| Optimizer | Adam |
| Initial learning rate | `1e-4` |
| Loss | Categorical Crossentropy |
| Output activation | Softmax |
| Validation split | `20%` |
| Early stopping | Patience = 5 |
| LR reduction | Factor = 0.2 |

---

## 🖼️ Image Preprocessing

Several preprocessing steps are applied before training:

### 1. Hand Detection ✋

MediaPipe Hands is used to locate the hand in each image.

If the hand is detected, a bounding box is created around the detected hand with additional padding.

### 2. Fallback Crop 🔍

If MediaPipe cannot detect a hand, the notebook uses a center crop as a fallback.

From the 5,280 images:

- Total images processed: **5,280**
- Fallback crops: **500**

### 3. Image Processing

The cropped image is:

```text
Crop
 ↓
Random background augmentation
 ↓
Grayscale
 ↓
Gaussian blur
 ↓
Convert back to RGB
 ↓
Resize to 224 × 224
 ↓
MobileNet preprocess_input
```

---

## 🔄 Data Augmentation

During training, `ImageDataGenerator` applies several transformations:

- Rotation: ±20°
- Width shift: 15%
- Height shift: 15%
- Zoom: 20%
- Brightness: 0.5–1.5
- Shear: 20%
- Horizontal flip
- Nearest-pixel filling

These transformations are intended to expose the model to variations in image appearance during training.

---

## 📊 Dataset

The notebook uses the **SIBI dataset** available through Kaggle.

The dataset is expected at:

```python
/kaggle/input/sibi-dataset/SIBI
```

with a directory structure similar to:

```text
SIBI/
├── A/
├── B/
├── C/
├── D/
├── ...
├── W/
├── X/
└── Y/
```

The dataset itself is **not included in this repository**.

---

## 📈 Results

The dataset was split using a stratified **80:20 train-validation split**.

### Validation Performance

| Metric | Result |
|---|---:|
| Validation images | 1,056 |
| Validation accuracy | **98%** |
| Macro average AUC | **0.9999** |
| Macro precision | **0.98** |
| Macro recall | **0.98** |
| Macro F1-score | **0.98** |

The classification report shows strong performance across the 24 classes, with individual F1-scores ranging from approximately **0.94 to 1.00**.

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming language |
| 🧠 TensorFlow / Keras | Deep learning & model training |
| 📱 MobileNet | Image classification backbone |
| ✋ MediaPipe | Hand detection |
| 👁️ OpenCV | Image processing |
| 🔢 NumPy | Numerical computation |
| 📊 Scikit-learn | Data splitting & evaluation |
| 📈 Matplotlib | Visualization |
| 🎨 Seaborn | Visualization |
| ⏳ tqdm | Progress tracking |
| ☁️ Kaggle | Notebook & dataset environment |

---

## 🚀 How to Run

### 1. Open the notebook

The main project is provided as:

```text
Sign-Language-Detection.ipynb
```

It was developed in a **Kaggle Notebook** environment.

### 2. Install the required MediaPipe dependencies

```bash
pip install mediapipe==0.10.7 protobuf==3.20.3
```

### 3. Prepare the dataset

Place the SIBI dataset in the expected Kaggle directory:

```text
/kaggle/input/sibi-dataset/SIBI
```

### 4. Run the notebook

Run the notebook cells sequentially:

```text
Library Setup
      ↓
Dataset Preparation
      ↓
Hand Detection
      ↓
Image Preprocessing
      ↓
Train/Validation Split
      ↓
Augmentation
      ↓
MobileNet Training
      ↓
Evaluation
```

---

## 📁 Repository Structure

```text
📦 Sign-Language-Detection-Model-using-MobileNet-98Accuracy
│
├── 📓 Sign-Language-Detection.ipynb
└── 📄 README.md
```

---

## 📌 Notes

- This project focuses on **image classification of SIBI hand-sign images**.
- The notebook uses **MediaPipe Hands** for hand localization before classification.
- The reported 98% accuracy is the validation accuracy obtained in the notebook; it should not be interpreted as guaranteed real-world performance.
- The project does **not** include a deployed real-time application or a webcam inference interface.
- The dataset is not included in this repository.

---

## 👩‍💻 Project

**Sign Language Detection Model using MobileNet**

Built as an academic / machine learning project to explore **computer vision, image preprocessing, transfer learning, and deep learning-based classification**.

### 🔖 Topics

`Computer Vision` · `Deep Learning` · `Image Classification` · `Transfer Learning` · `MobileNet` · `TensorFlow` · `MediaPipe` · `SIBI`

---

⭐ If you find this project useful, feel free to explore the notebook and its implementation!
