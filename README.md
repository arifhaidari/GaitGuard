# Gait Phase Analysis Project

## Overview

This project focuses on **gait phase analysis** using machine learning and deep learning techniques to classify walking conditions (e.g., slow, medium, and fast walking) and predict gait phases. The force data used in this project was captured from a **split-belt treadmill** synchronized with a motion capture system.

### Applications:

- **Rehabilitation Monitoring**: Track recovery progress in patients with gait disorders.
- **Gait Abnormality Detection**: Identify deviations in normal walking patterns.
- **Performance Assessment**: Analyze gait in athletes for training optimization.

This project leverages **classification and regression algorithms** such as **MLPClassifier**, **LSTM**, and **Deep Neural Networks (DNNs)**. The final trained **MLPClassifier** was selected as the best-performing model and is used for deployment and prediction.

---

## Key Features

### Dataset Overview

The dataset was sourced from the **GaitPhase Database** and includes data collected using a **Qualisys motion capture system** and an **instrumented split-belt treadmill**. The dataset contains 756 files (~600 MB), with a subset used for training due to hardware limitations.

1. **Dataset used for initial experiments (classification and regression)**:

   - Files: `GP1_0.6_force.csv`, `GP1_0.6_marker.csv`

2. **Dataset used for MLPClassifier and Deep Learning**:

   - Files: `GP1_0.6_force.csv`, `GP1_0.7_force.csv`, `GP1_0.8_force.csv`

3. **Key Data Characteristics**:
   - Over 30 features, including ground reaction forces in x, y, z axes for two force plates.
   - Walking conditions represented as labels (`Slow`, `Medium`, `Fast`).

---

## Problem Statement

### Objectives:

1. **Classification**:
   - Classify walking speeds (`Slow`, `Medium`, `Fast`) based on force data.
2. **Regression**:
   - Predict numerical gait parameters such as step duration or force magnitude.

---

## Models and Techniques

1. **Machine Learning Models**:

   - **MLPClassifier**:
     - Achieved the highest accuracy (~77%).
     - Hyperparameters:
       - Solver: `adam`
       - Hidden Layers: `(200, 100, 55)`
       - Activation: `tanh`
       - Max Iterations: `50`

2. **Deep Learning Models**:

   - **LSTM**:
     - Used for time-series data to analyze temporal dependencies.
     - Accuracy: ~33% (due to limited time-step configuration).
   - **DNN**:
     - Accuracy: ~72%
     - Achieved moderate results compared to MLPClassifier.

3. **Deployment**:
   - Saved the trained MLPClassifier model using `joblib`.
   - Recreated preprocessing pipelines for new data predictions.

---

## Project Structure

```
├── data/
│   ├── GP1_0.6_force.csv
│   ├── GP1_0.7_force.csv
│   ├── GP1_0.8_force.csv
├── notebooks/
│   ├── preprocessing_and_training.ipynb
│   ├── prediction.ipynb
├── models/
│   ├── mlp_classifier_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
├── README.md
└── requirements.txt
```

---

## Steps Followed

### 1. Data Import and Preprocessing

- Imported datasets representing walking speeds (`Slow`, `Medium`, `Fast`).
- Combined the datasets and scaled features using `StandardScaler`.

**Code Snippet**:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

---

### 2. Exploratory Data Analysis (EDA)

- Visualized feature distributions and label balance using `seaborn`.
- Correlation heatmaps to identify relationships between features.

---

### 3. Model Training

- **MLPClassifier**:
  - Achieved ~77% accuracy.
  - Chosen as the final model due to its superior performance and simplicity.
- **LSTM**:

  - Used for time-series analysis but underperformed (~33% accuracy).

- **DNN**:
  - Achieved ~72% accuracy, moderately competitive.

---

### 4. Model Evaluation

**Metrics Used**:

- Accuracy, Precision, Recall, F1-Score.
- Confusion Matrix for class-wise performance.

---

### 5. Model Deployment

- **Saved the model**:

  ```python
  import joblib
  joblib.dump(mlp, "mlp_classifier_model.joblib")
  ```

- **Used for predictions**:

  ```python
  # Load the model
  loaded_model = joblib.load("mlp_classifier_model.joblib")

  # Scale new data
  new_data_scaled = scaler.transform(new_data)

  # Predict
  predictions = loaded_model.predict(new_data_scaled)
  ```

---

## Results and Insights

- **MLPClassifier** performed the best, demonstrating the effectiveness of neural networks in analyzing force data.
- **Deep Learning models (DNNs, LSTM)** require more extensive time-series data and configurations for improvement.

---

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/arifhaidari/gait_phase_analysis.git
   cd gait_phase_analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Future Work

- Extend the project to predict specific gait phases (`Stance`, `Swing`, etc.).
- Incorporate additional sensor data (e.g., accelerometer, gyroscope).
- Deploy the trained model as an API using Flask or FastAPI.
- Optimize hyperparameters for deep learning models.

---

## Acknowledgments

This project was developed using:

- [GaitPhase Database](https://www.mad.tf.fau.de/research/activitynet/gaitphase-database/)
- Domain knowledge, research papers, and external references:
  - [Gait Disorders](https://my.clevelandclinic.org/health/diseases/21092-gait-disorders)
  - [Gait Analysis - PubMed](https://pubmed.ncbi.nlm.nih.gov/15519595/)
  - [Additional Research](https://drive.google.com/file/d/1gC5iiZM9-A_a0_9x29eDBw9qMCljv4Vy/view)
  - Developed with guidance and context provided through interactions with **ChatGPT**.
