# Brain Tumor Classification using KNN + PCA

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
