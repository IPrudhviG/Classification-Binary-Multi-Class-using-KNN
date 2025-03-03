# 1️⃣ Traffic Sign Classification using K-Nearest Neighbors (KNN)

## Overview
This project implements a **Traffic Sign Classification model** using the **K-Nearest Neighbors (KNN) algorithm**. The dataset used is the **German Traffic Sign Recognition Benchmark (GTSRB)**, which contains images of various traffic signs along with their class labels. The project includes:

- **Feature Extraction** using **Histogram of Oriented Gradients (HOG)**
- **Dimensionality Reduction** using **Principal Component Analysis (PCA)**
- **Handling Imbalanced Data** using **SMOTE**
- **Hyperparameter Tuning** using **GridSearchCV**
- **Performance Comparison** for different values of `k`

---

## Dataset
The dataset contains images of **traffic signs** and their corresponding **Class IDs**. The data is provided in two files:
- `Train.csv`: Contains file paths and labels for training images.
- `Test.csv`: Contains file paths for testing images.

Each image belongs to one of **43 traffic sign classes**. The dataset can be found [here](https://benchmark.ini.rub.de/gtsrb_news.html).

---

## Project Workflow

### 1️⃣ Data Preprocessing
- **Loaded images** and applied **grayscale conversion**.
- **Resized images** to **32x32 pixels**.
- **Extracted HOG features** to obtain meaningful feature vectors.
- **Standardized features** using **StandardScaler**.

### 2️⃣ Dimensionality Reduction
- Applied **PCA** to reduce feature dimensionality while retaining variance.

### 3️⃣ Handling Imbalanced Data
- Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.

### 4️⃣ Training the KNN Model
- **GridSearchCV** was used to find the best `k` value.
- However, after observing that the **ROC-AUC score was not optimal** for `k=3`, **manual tuning** was done by increasing `k=9`, leading to improved performance.

### 5️⃣ Model Evaluation & Performance Comparison
- **Accuracy Score**
- **Confusion Matrix**
- **ROC-AUC Score**
- **Misclassification Analysis**

---

## Installation & Dependencies
To run the project, install the required Python libraries:
```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn imbalanced-learn joblib




#Brain Tumor Classification using KNN + PCA

## Overview
This project aims to classify brain tumors using a **K-Nearest Neighbors (KNN) classifier** with **Principal Component Analysis (PCA)** for dimensionality reduction. The dataset consists of MRI brain scan features, and the model predicts whether a given scan belongs to a tumor class.

## Dataset
- The dataset contains extracted features from brain MRI images.
- The dataset was preprocessed by normalizing the features using **StandardScaler**.
- **SMOTE** was applied to handle class imbalance.

## Project Workflow
1. **Data Loading:** Read the dataset and inspect its structure.
2. **Preprocessing:**  
   - Normalize the data using `StandardScaler`.  
   - Apply **PCA** to reduce dimensions (10 principal components).
3. **Handling Imbalanced Data:**  
   - Use **SMOTE** to balance class distribution.
4. **Model Training:**  
   - Train a **KNN classifier** with **Euclidean distance**.
5. **Model Evaluation:**  
   - Compute **Confusion Matrix**, **Classification Report**, **ROC-AUC Score**, **Balanced Accuracy**, and **Precision-Recall AUC**.
6. **Visualization:**  
   - Heatmap of the confusion matrix.  
   - PCA scatter plot.  
   - Class distribution histogram.  
   - ROC and Precision-Recall curves.

## Results
- **ROC-AUC Score:** **0.9847**  
- **Balanced Accuracy:** **0.9847**  
- **Precision-Recall AUC:** **0.9911**  

The model demonstrates **high classification performance**, making KNN + PCA an effective approach for this dataset.

## Future Improvements
- **Hyperparameter tuning** for optimal KNN performance.
- Experiment with different **distance metrics** (e.g., Manhattan, Minkowski).
- **Compare with other models** such as SVM, Decision Trees, or CNNs for feature extraction.

## Installation & Usage
### Prerequisites
Ensure you have Python installed with the following libraries:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
