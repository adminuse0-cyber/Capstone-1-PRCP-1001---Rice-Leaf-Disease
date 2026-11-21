# PRCP-1001: Rice Leaf Disease Detection & Classification[cite: 1]

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Deliverables](#3-project-deliverables)
4. [Dataset Architecture & Automated Ingestion](#4-dataset-architecture--automated-ingestion)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Methodology & Modeling Pipeline](#6-methodology--modeling-pipeline)
7. [Model Comparison & Results](#7-model-comparison--results)
8. [Challenges Faced & Engineering Solutions](#8-challenges-faced--engineering-solutions)
9. [Tech Stack](#9-tech-stack)
10. [Setup & Usage](#10-setup--usage)

---

## 1. Project Overview

**PRCP-1001** is an end-to-end Computer Vision, Machine Learning, and Deep Learning project designed to automatically detect and classify three major foliar diseases affecting rice (*Oryza sativa*) plants from leaf images[cite: 1]. All stages of the workflow—automated archive extraction, exploratory data analysis, classical ML baselines, custom CNN training, pretrained transfer learning, controlled data augmentation experiments, and production deployment analysis—are implemented inside a single reproducible Jupyter Notebook (`Rice Leaf Disease Detection.ipynb`)[cite: 1].

---

## 2. Problem Statement

Rice (*Oryza sativa*) is a staple food crop for more than half of the world's population, and foliar diseases can cause severe yield losses if not detected early[cite: 1]. Manual inspection in agricultural fields is slow and error-prone. This project builds and evaluates automated classification models to distinguish between three rice leaf disease classes[cite: 1]:

1. **Bacterial Leaf Blight** — caused by *Xanthomonas oryzae* pv. *oryzae*[cite: 1]
2. **Brown Spot** — caused by *Bipolaris oryzae*[cite: 1]
3. **Leaf Smut** — caused by *Entyloma oryzae*[cite: 1]

---

## 3. Project Deliverables

* **Task 1 — Complete Exploratory Data Analysis (EDA) Report:** Automated extraction of nested class ZIP archives from `datasets/Data`[cite: 1], class balance verification[cite: 1], resolution and aspect-ratio profiling[cite: 1], RGB color channel intensity analysis[cite: 1], visual inspection galleries[cite: 1], and statistical summaries[cite: 1].
* **Task 2 — Multi-Model Disease Classification:** Implementation and evaluation of feature-engineered classical ML models (**SVM** and **Random Forest**)[cite: 1], a **Custom CNN** built from scratch[cite: 1], and pretrained Transfer Learning architectures (**MobileNetV2**, **ResNet50**, and **EfficientNetB0**)[cite: 1].
* **Task 3 — Data Augmentation & Regularization Analysis:** Exploration and visualization of geometric and photometric image augmentation techniques[cite: 1], followed by a controlled experiment benchmarking performance **without vs. with augmentation**[cite: 1].
* **Model Comparison Report & Production Recommendation:** Quantitative comparison across **Accuracy, Precision, Recall, F1-Score, Model Size, and Inference Speed** to recommend the optimal model for real-world agricultural deployment[cite: 1].
* **Report on Challenges Faced & Engineering Solutions:** Documentation of dataset anomalies (nested ZIP archives, 119 vs. 120 sample count, mixed `.JPG`/`.jpg` extensions, extreme aspect ratios, and small sample size) alongside the exact engineering solutions applied[cite: 1].

---

## 4. Dataset Architecture & Automated Ingestion

### Directory Layout

The raw dataset stores each disease class inside an individual `.zip` archive located in `datasets/Data`[cite: 1]:

```text
Rice-Leaf-Disease-Detection/
├── datasets/
│   ├── Data/
│   │   ├── Bacterial leaf blight-20200814T055237Z-001.zip
│   │   ├── Brown spot-20200814T055208Z-001.zip
│   │   └── Leaf smut-20200814T055530Z-001.zip
│   └── extracted_data/                  # Auto-generated on notebook execution
│       ├── Bacterial leaf blight/
│       ├── Brown spot/
│       └── Leaf smut/
├── Rice Leaf Disease Detection.ipynb
└── README.md
```

### Automated Extraction Pipeline

The `prepare_rice_leaf_dataset()` setup utility ensures reproducible, non-destructive data loading[cite: 1]:

1. Automatically resolves `datasets/Data` relative to the notebook directory (with an absolute Windows path fallback)[cite: 1].
2. Extracts all `.zip` archives cleanly into `datasets/extracted_data` while preserving the original compressed files[cite: 1].
3. Falls back directly to `datasets/Data` if class folders have already been manually extracted[cite: 1].
4. Verifies and sorts all detected class directories (`['Bacterial leaf blight', 'Brown spot', 'Leaf smut']`)[cite: 1].

---

## 5. Exploratory Data Analysis (EDA)

A custom metadata profiling script scans all valid image files (`.jpg`, `.jpeg`, `.png`, matched case-insensitively) to build a structured DataFrame (`df_eda`) capturing spatial dimensions, aspect ratios, megapixels, file sizes (KB), color modes, and per-channel mean RGB values[cite: 1].

### Dataset Overview

* **Total Valid Images:** `119` images across 3 classes[cite: 1]
* **File Extension Breakdown:** `67` uppercase `.JPG` files and `52` lowercase `.jpg` files[cite: 1]
* **Target Input Size:** Images are profiled at raw resolution before resizing to `224 × 224` for deep learning pipelines[cite: 1]

### Class-Wise Statistical Summary

| Disease Class | Image Count | Min Width (px) | Max Width (px) | Mean Width (px) | Min Height (px) | Max Height (px) | Mean Height (px) | Mean Aspect Ratio | Mean Size (KB) | Avg R | Avg G | Avg B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bacterial leaf blight**[cite: 1] | 40[cite: 1] | 3081[cite: 1] | 3081[cite: 1] | 3081.0[cite: 1] | 897[cite: 1] | 897[cite: 1] | 897.0[cite: 1] | 3.43[cite: 1] | 432.9[cite: 1] | 193.1[cite: 1] | 188.7[cite: 1] | 153.1[cite: 1] |
| **Brown spot**[cite: 1] | 40[cite: 1] | 286[cite: 1] | 3081[cite: 1] | 1951.3[cite: 1] | 88[cite: 1] | 900[cite: 1] | 589.2[cite: 1] | 3.28[cite: 1] | 240.2[cite: 1] | 191.4[cite: 1] | 188.9[cite: 1] | 127.0[cite: 1] |
| **Leaf smut**[cite: 1] | 39[cite: 1] | 250[cite: 1] | 3081[cite: 1] | 2111.8[cite: 1] | 71[cite: 1] | 897[cite: 1] | 635.2[cite: 1] | 3.25[cite: 1] | 248.7[cite: 1] | 180.3[cite: 1] | 181.6[cite: 1] | 141.7[cite: 1] |

### Key Analytical Findings

* **Near-Balanced Class Distribution:** The dataset contains `40` *Bacterial leaf blight*, `40` *Brown spot*, and `39` *Leaf smut* images (`119` total instead of `120`)[cite: 1].
* **Wide Horizontal Aspect Ratios:** Mean aspect ratios span `3.25` to `3.43` across all classes due to the elongated shape of rice leaves[cite: 1].
* **Resolution Variance Across Classes:** All *Bacterial leaf blight* samples share a uniform high resolution (`3081 × 897`, averaging `432.9 KB`)[cite: 1], while *Brown spot* (`286 × 88` to `3081 × 900`) and *Leaf smut* (`250 × 71` to `3081 × 897`) show substantial variance in both pixel dimensions and file sizes[cite: 1].
* **Chromatic Channel Differences:** *Brown spot* shows a noticeably lower mean Blue intensity (`127.0`) than *Bacterial leaf blight* (`153.1`) and *Leaf smut* (`141.7`), while *Leaf smut* exhibits lower mean Red (`180.3`) and Green (`181.6`) intensities[cite: 1].
* **Visualizations Included:** Class count bar chart, file size (KB) boxplot, width-vs-height resolution scatter plot, RGB intensity distributions, and class sample inspection galleries[cite: 1].

---

## 6. Methodology & Modeling Pipeline

### Reproducibility & Evaluation Setup

* Global random seed fixed at `SEED = 42` across `random`, `numpy`, and `tensorflow`[cite: 1].
* Feature scaling and splitting handled via `StandardScaler` and `train_test_split` from `scikit-learn`[cite: 1].
* Evaluation performed using `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`, and `confusion_matrix`[cite: 1].

### Architectures Implemented

1. **Classical Machine Learning (Feature-Engineered)**[cite: 1]
   * **Random Forest Classifier (`RandomForestClassifier`)**[cite: 1]
   * **Support Vector Machine (`SVC`)**[cite: 1]
2. **Custom Deep Learning Baseline**[cite: 1]
   * **Custom CNN** built from scratch using `tensorflow.keras` (`layers`, `models`)[cite: 1]
3. **Transfer Learning (Pretrained Backbones)**[cite: 1]
   * **MobileNetV2**[cite: 1]
   * **ResNet50**[cite: 1]
   * **EfficientNetB0**[cite: 1]

### Data Augmentation & Regularization Study

* Applies geometric and photometric image augmentations to address the small dataset size (`119` images)[cite: 1].
* Includes visual demonstrations of augmented outputs and a controlled **without vs. with augmentation** benchmark[cite: 1].

---

## 7. Model Comparison & Results

> Replace the placeholder values below with the final numbers printed in your notebook's comparison report cells.

### Quantitative Benchmark

| Model | Type | Accuracy | Precision | Recall | F1-Score | Model Size (MB) | Inference Speed (ms/img) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest**[cite: 1] | Classical ML[cite: 1] | -- | -- | -- | -- | -- | -- |
| **SVM (SVC)**[cite: 1] | Classical ML[cite: 1] | -- | -- | -- | -- | -- | -- |
| **Custom CNN**[cite: 1] | Deep Learning (Scratch)[cite: 1] | -- | -- | -- | -- | -- | -- |
| **MobileNetV2**[cite: 1] | Transfer Learning[cite: 1] | -- | -- | -- | -- | -- | -- |
| **ResNet50**[cite: 1] | Transfer Learning[cite: 1] | -- | -- | -- | -- | -- | -- |
| **EfficientNetB0**[cite: 1] | Transfer Learning[cite: 1] | -- | -- | -- | -- | -- | -- |

### Augmentation Experiment (Without vs. With Augmentation)

| Setup | Accuracy | F1-Score | Observation |
| :--- | :---: | :---: | :--- |
| **Without Augmentation**[cite: 1] | -- | -- | Baseline performance without regularization |
| **With Augmentation**[cite: 1] | -- | -- | Regularized performance using geometric & photometric transforms[cite: 1] |

### Production Recommendation

* **Best Model for Deployment:** `[Add recommended model name]`
* **Rationale:** `[Add justification based on Accuracy, F1-Score, Model Size, and Inference Speed]`[cite: 1]

---

## 8. Challenges Faced & Engineering Solutions

| Challenge | Description | Solution Implemented |
| :--- | :--- | :--- |
| **Nested Class ZIP Archives**[cite: 1] | Class images are stored inside three timestamped `.zip` files within `datasets/Data`[cite: 1]. | Implemented `prepare_rice_leaf_dataset()` using `zipfile` and `pathlib` to unpack into `datasets/extracted_data` without modifying original ZIPs[cite: 1]. |
| **Dataset Count Anomaly (119 vs. 120)**[cite: 1] | *Leaf smut* contains `39` images rather than `40`, totaling `119` valid images[cite: 1]. | Dynamically audited all files using `df_eda` rather than relying on hardcoded sample counts[cite: 1]. |
| **Mixed File Extension Casing**[cite: 1] | Files mix `.JPG` (`67` images) and `.jpg` (`52` images)[cite: 1]. | Enforced case-insensitive extension filtering (`img_path.suffix.lower()`)[cite: 1]. |
| **Extreme Aspect Ratios & Resolution Variance**[cite: 1] | Aspect ratios average `3.25–3.43` and resolutions range from `250 × 71` up to `3081 × 900`[cite: 1]. | Profiled resolution distributions during EDA before resizing inputs to `224 × 224`[cite: 1]. |
| **Small Sample Size (`N = 119`)**[cite: 1] | Small training data increases overfitting risk for deep CNNs[cite: 1]. | Evaluated Transfer Learning (**MobileNetV2**, **ResNet50**, **EfficientNetB0**), classical ML models, and geometric/photometric data augmentation[cite: 1]. |
| **Native Windows GPU Support**[cite: 1] | TensorFlow `>= 2.11` (`2.22.0-rc1`) does not support native Windows CUDA/cuDNN GPUs (`GPU Available: False`)[cite: 1]. | Validated CPU execution workflow; use WSL2 or the `TensorFlow-DirectML` plugin for local Windows GPU acceleration[cite: 1]. |

---

## 9. Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow (`2.22.0-rc1`) & Keras (`tensorflow.keras`)[cite: 1]
* **Classical Machine Learning:** `scikit-learn` (`SVC`, `RandomForestClassifier`, `StandardScaler`, `train_test_split`, evaluation metrics)[cite: 1]
* **Image Processing:** OpenCV (`cv2`), Pillow (`PIL`)[cite: 1]
* **Data Analysis & Visualization:** `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`[cite: 1]
* **Utilities:** `pickle`, `zipfile`, `pathlib`[cite: 1]

---

## 10. Setup & Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn pillow opencv-python scikit-learn tensorflow jupyter
```

### 3. Prepare the Dataset

Place the three raw class `.zip` files (or already-extracted class folders) inside `datasets/Data/`[cite: 1].

### 4. Run the Notebook

```bash
jupyter notebook "Rice Leaf Disease Detection.ipynb"
```

Run all cells sequentially to automatically extract the dataset, execute the full EDA pipeline, train and evaluate all models, run the augmentation study, and generate the comparison report[cite: 1].
