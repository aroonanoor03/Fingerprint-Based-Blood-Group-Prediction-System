# Fingerprint Based Blood Group Prediction

An end-to-end Machine Learning and Deep Learning system designed to predict blood groups from fingerprint scan images. The pipeline is implemented in Python, utilizing image preprocessing, feature extraction, and multiple classification algorithms (Support Vector Machines, Random Forests, and VGG16 Convolutional Neural Networks).

## Project Overview

Fingerprints are unique, lifelong biological patterns. This project investigates the correlation between fingerprint ridge configurations and human blood groups.

### Features
1. **Dataset Loading & Exploratory Data Analysis (EDA)**: Scans 6,000 images, checks class distributions, and handles image size variations.
2. **Image Preprocessing**: Applies grayscale conversion, standardizes sizes to 224x224, removes noise using a 3x3 median filter, enhances ridge contrast with histogram equalization, and normalizes intensity to [0,1].
3. **Feature Extraction**: Extracts high-dimensional shape descriptors using Histogram of Oriented Gradients (HOG).
4. **Machine Learning Classifiers**:
   - **Linear Support Vector Machine (SVM)** (Best performing: **90.58% accuracy**)
   - **Random Forest** (88.00% accuracy)
5. **Deep Learning Classifier**:
   - **VGG16 Transfer Learning** CNN (trained Classifier Head on CPU: 47.92% accuracy)
6. **Robust Validation Guard**: Automatically analyzes raw pixel block-level variances to reject non-fingerprint uploads (like logos, text, or shapes) in real-time.
7. **Interactive Web Dashboard**: Beautiful dark-themed Streamlit GUI showing the prediction result and interactive Plotly confidence bar charts.

---

## Installation & Setup

1. **Clone or copy the project files** to your local workspace.
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Place the dataset**: Ensure the folder `dataset_blood_group` containing subfolders for each blood group class (`A+`, `A-`, `AB+`, `AB-`, `B+`, `B-`, `O+`, `O-`) is in the project root directory.

---

## How to Run

### 1. Exploratory Data Analysis (EDA)
Scan the dataset, print statistics, and generate sample grids:
```bash
python main.py
```

### 2. Train Models
Execute the training scripts to train the classifiers and save the trained weights:
- **Train SVM**:
  ```bash
  python -m src.svm_model
  ```
- **Train Random Forest**:
  ```bash
  python -m src.random_forest_model
  ```
- **Train VGG16 CNN**:
  ```bash
  python -m src.cnn_model
  ```

### 3. Model Comparison
Compare the accuracy, precision, and recall of the trained models and write reports to `results/`:
```bash
python -m src.evaluation
```

### 4. CLI Inference
Run prediction on a single, custom fingerprint scan file:
```bash
python -m src.predict path/to/your/fingerprint.bmp
```

### 5. Start Streamlit Web Dashboard
Launch the interactive web interface:
```bash
streamlit run app.py
```

---

## File Structure

```text
├── dataset_blood_group/        # Original fingerprint BMP images (grouped by class)
├── models/                     # Saved trained models (.pkl and .pth)
├── results/                    # Saved confusion matrices and evaluation reports
├── src/                        # Code implementation packages
│   ├── __init__.py
│   ├── cnn_model.py            # VGG16 training and evaluation pipeline
│   ├── evaluation.py           # Classifier comparison and report generator
│   ├── feature_extraction.py   # HOG feature extractor helpers
│   ├── predict.py              # CLI inference engine
│   ├── preprocessing.py        # Ridge enhancement and filter operations
│   ├── random_forest_model.py  # Random Forest training pipeline
│   └── svm_model.py            # Support Vector Machine training pipeline
├── app.py                      # Streamlit interactive GUI dashboard
├── main.py                     # Exploratory Data Analysis entrypoint
├── requirements.txt            # Package dependencies
└── README.md                   # Project documentation
```

---

## System Analysis & Summary

| Classifier | Test Accuracy | Precision (Macro) | Recall (Macro) |
| :--- | :--- | :--- | :--- |
| **Linear SVM (HOG)** | **90.58%** | **90.80%** | **90.52%** |
| **Random Forest (HOG)** | **88.00%** | **88.43%** | **87.84%** |
| **VGG16 CNN** | **47.92%** | **56.24%** | **46.79%** |

* Handcrafted features extracted via HOG combined with a **Linear SVM** yield the highest performance because orientation gradients directly model ridge directions.
* Transfer learning with frozen feature extractors performs moderately on fingerprints on CPU because VGG16's ImageNet features (optimized for natural color photos) do not naturally capture subtle fingerprint textures without early layer fine-tuning.
