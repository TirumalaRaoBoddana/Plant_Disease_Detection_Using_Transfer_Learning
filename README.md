# 🌿 Plant Disease Detection Using Transfer Learning

> A deep learning-powered web application that detects and classifies **38 plant diseases** across **14 crop species** from leaf images — built with ResNet50 transfer learning and deployed via a Streamlit interface.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Solution Approach](#-solution-approach)
- [Tech Stack](#-tech-stack)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Supported Plants & Diseases](#-supported-plants--diseases)
- [Project Structure](#-project-structure)
- [How to Run Locally](#-how-to-run-locally)
- [Application Workflow](#-application-workflow)
- [Key Results & Metrics](#-key-results--metrics)
- [Skills Demonstrated](#-skills-demonstrated)
- [Author](#-author)

---

## 🧭 Project Overview

Plant diseases cause significant agricultural losses worldwide, often going undetected until visible damage is severe. Early and accurate identification is critical for timely intervention. This project presents an **end-to-end deep learning solution** that:

- Accepts a **leaf image** as input
- Runs it through a fine-tuned **ResNet50 convolutional neural network**
- Outputs the **predicted disease name** along with a **confidence score**
- Warns the user when prediction confidence is low (< 60%), preventing misuse on out-of-distribution images

The model was trained on the **PlantVillage dataset** — one of the largest open-source plant pathology datasets — and deployed as a browser-accessible web application using **Streamlit**.

---

## ❗ Problem Statement

- Farmers and agronomists often lack rapid, accessible tools to diagnose crop diseases in the field
- Manual inspection is time-consuming, error-prone, and requires domain expertise
- Delayed detection leads to crop loss, excessive pesticide use, and economic damage
- **Goal:** Build a scalable, accurate AI model that can classify plant leaf diseases from a photograph with high accuracy and low latency

---

## 💡 Solution Approach

| Step | Action |
|------|--------|
| 1 | Loaded and preprocessed the PlantVillage dataset (224×224 RGB images) |
| 2 | Applied **Transfer Learning** using ResNet50 pre-trained on ImageNet |
| 3 | Fine-tuned top layers with custom dense classification head for 38 classes |
| 4 | Evaluated model performance using accuracy, loss curves, and classification reports |
| 5 | Saved trained model as `.keras` format with class index mapping in JSON |
| 6 | Built an interactive **Streamlit web application** for real-time inference |
| 7 | Added confidence thresholding to flag uncertain predictions |

---

## 🛠 Tech Stack

| Category | Tools & Libraries |
|----------|------------------|
| **Programming Language** | Python 3.11+ |
| **Deep Learning Framework** | TensorFlow 2.20, Keras 3.10 |
| **Transfer Learning Model** | ResNet50 (ImageNet pre-trained) |
| **Data Processing** | NumPy, Pillow (PIL) |
| **Model Evaluation** | Scikit-learn, Matplotlib, Seaborn |
| **Web Application** | Streamlit 1.32+ |
| **Development Environment** | Jupyter Notebook, Google Colab |
| **Version Control** | Git, GitHub |

---

## 🧠 Model Architecture

```
Input Layer       →  224 × 224 × 3 (RGB leaf image)
        ↓
ResNet50 Base     →  Pre-trained on ImageNet (frozen base layers)
  (50 layers)        Extracts hierarchical visual features
        ↓
Global Average    →  Reduces spatial dimensions to 1D feature vector
  Pooling
        ↓
Dense Layer       →  Fully connected layer with ReLU activation
        ↓
Dropout Layer     →  Regularization to prevent overfitting
        ↓
Output Layer      →  Dense(38) with Softmax activation
                     → Probability distribution over 38 disease classes
```

**Why ResNet50?**
- Residual connections solve the vanishing gradient problem in deep networks
- Pre-trained ImageNet weights give a powerful feature extractor head-start
- Significantly reduces training time and data requirements vs. training from scratch
- Proven performance on image classification benchmarks

---

## 📦 Dataset

- **Name:** PlantVillage Dataset
- **Source:** [Kaggle / PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Total Classes:** 38 (disease + healthy categories)
- **Crop Species Covered:** 14
- **Image Format:** RGB, resized to 224 × 224 pixels
- **Preprocessing:** ResNet50-specific normalization via `preprocess_input()`

---

## 🌾 Supported Plants & Diseases

The model classifies **38 categories** across the following crops:

| Crop | Conditions Detected |
|------|-------------------|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Bell Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 📁 Project Structure

```
Plant_Disease_Detection_Using_Transfer_Learning/
│
├── plant-disease-detection.ipynb   # Full model training pipeline (EDA → Training → Evaluation)
├── app.py                          # Streamlit web application for real-time inference
├── final_resnet50_model.keras      # Trained ResNet50 model weights (saved in Keras format)
├── class_indices.json              # JSON mapping: class name → integer label index
├── requirements.txt                # Python dependencies with pinned versions
└── plant_disease/                  # Dataset directory (images organized by class folder)
```

---

## ▶ How to Run Locally

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/TirumalaRaoBoddana/Plant_Disease_Detection_Using_Transfer_Learning.git
cd Plant_Disease_Detection_Using_Transfer_Learning
```

### Step 2 — Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the Streamlit Application
```bash
streamlit run app.py
```

### Step 5 — Use the App
- Open your browser and go to `http://localhost:8501`
- Upload a clear leaf image (JPG, JPEG, or PNG)
- Click **"Predict Disease"**
- View the predicted disease name and confidence score

> **Note:** For best results, use well-lit, close-up images of individual leaves against a plain background, similar to the training data.

---

## 🔄 Application Workflow

```
User uploads leaf image (JPG/PNG)
            ↓
Image resized to 224 × 224 pixels
            ↓
ResNet50 preprocess_input() normalization applied
            ↓
Model predicts probability scores for all 38 classes
            ↓
Class with highest probability selected (argmax)
            ↓
Confidence score extracted
            ↓
  [Confidence ≥ 60%]          [Confidence < 60%]
  Show: Disease Name       →   Show: Disease Name
        + Confidence %           + Low Confidence Warning
```

---

## 📊 Key Results & Metrics

| Metric | Value |
|--------|-------|
| **Model Architecture** | ResNet50 (Transfer Learning) |
| **Total Disease Classes** | 38 |
| **Validation Accuracy** | > 90% |
| **Confidence Thresholding** | < 60% triggers out-of-distribution warning |
| **Inference Latency** | Real-time (< 2 seconds per image on CPU) |
| **Model Size** | Saved as `.keras` format for efficient loading |
| **Model Caching** | `@st.cache_resource` used for fast repeated predictions |

---

## 💼 Skills Demonstrated

This project showcases a complete machine learning engineering workflow relevant to software engineering and AI roles:

- **Deep Learning:** Designed and fine-tuned a ResNet50 CNN architecture using TensorFlow/Keras for multi-class image classification
- **Transfer Learning:** Applied ImageNet pre-trained weights, significantly reducing training time while achieving high accuracy
- **Data Engineering:** Managed image preprocessing pipelines including resizing, normalization, and augmentation using NumPy and Pillow
- **Model Evaluation:** Assessed performance using validation accuracy, loss curves, and confidence-based thresholding for real-world reliability
- **Software Development:** Built a production-ready Streamlit web application with model caching, file upload handling, and clean UI/UX
- **Python Programming:** End-to-end Python implementation covering data loading, model training, serialization, and inference
- **Version Control:** Full codebase version-controlled and documented on GitHub for reproducibility and collaboration

---

## 👤 Author

**Boddana Tirumala Rao**
B.Tech Computer Science Engineering | Batch 2026
Rajiv Gandhi University of Knowledge Technologies, Nuzvid

- GitHub: [@TirumalaRaoBoddana](https://github.com/TirumalaRaoBoddana)
- LinkedIn: [tirumala-rao-boddana](https://www.linkedin.com/in/tirumala-rao-boddana-9b5a6b274/)
- Email: btirumalarao27@gmail.com

---

> *This project was built as part of an academic and professional portfolio to demonstrate practical deep learning and software engineering skills.*
