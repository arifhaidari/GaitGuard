# Gait Phase Analysis Project

## Overview

This project focuses on **gait phase analysis** using machine learning and deep learning techniques. The primary objective is to classify gait phases or walking conditions (e.g., slow, medium, and fast walking) using force data captured from a split-belt treadmill. This analysis is crucial in applications such as:

- Rehabilitation monitoring
- Detection of abnormal gait
- Performance assessment in athletes

The project leverages machine learning algorithms like **MLPClassifier (and many more classification and regression algorithms)** and advanced techniques such as **LSTM** and **Deep Neural Networks (DNNs)** to analyze and predict gait patterns. The final trained **MLPClassifier** model was saved and used for predictions, with detailed steps for preprocessing and deployment.

---

## Key Features

1. **Dataset**:

about dataset and sample of dataset:
the dataset that i have downloaded has 756 file and it is around 600 MB. I want to use small about of the dataset because data is just a fun project to learn to showcase my skills and also my processing power is very limited.

the dataset that I used for classification and regression:
files GP1_0.6_force.csv and GP1_0.6_marker.csv

dataset used for MLPClassifier neural network and deep learning:
GP1_0.6_force.csv
GP1_0.7_force.csv
GP1_0.8_force.csv

- Data collected using a Qualisys motion capture system and instrumented treadmill.
- Three datasets representing different walking speeds: `GP1_0.6_force.csv`, `GP1_0.7_force.csv`, and `GP1_0.8_force.csv`.
- Over 30 features including ground reaction forces (x, y, z components) for two force plates.

2. **Problem Solved**:

   - Classification of walking speeds (`Slow`, `Medium`, `Fast`).
   - Prediction of gait phases using force data.

3. **Models Used**:

   - Machine Learning:
     - **MLPClassifier** (best performing).
   - Deep Learning:
     - **LSTM** for time-series analysis.
     - **DNN** for comparison with traditional ML methods.

4. **Deployment**:
   - Saved the trained **MLPClassifier** for future predictions.
   - Detailed steps for loading and using the saved model.

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

### 1. **Data Import and Preprocessing**

- Imported datasets representing different walking speeds.
- Assigned meaningful labels (`Slow`, `Medium`, `Fast`) for classification.
- Combined and scaled the data using `StandardScaler`.

**Code Snippet**:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

---

### 2. **Exploratory Data Analysis (EDA)**

- Visualized feature distributions and label balance.
- Correlation heatmap to understand feature relationships.

**Visualizations**:

- Feature distributions using `seaborn`.
- Label distributions using bar plots.

---

### 3. **Model Training**

- **MLPClassifier**: Best-performing model with hyperparameters:
  - Solver: `adam`
  - Hidden Layers: `(200, 100, 55)`
  - Activation: `tanh`
  - Max Iterations: `50`

**Code Snippet**:

```python
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(200, 100, 55), activation='tanh', max_iter=50)
mlp.fit(X_train_scaled, y_train)
```

- **LSTM**:
  - Time-series model with 64 LSTM units and dropout regularization.
  - Used for analyzing temporal dependencies in force data.

**Code Snippet**:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

---

### 4. **Model Evaluation**

- **Metrics Used**:
  - Accuracy, Precision, Recall, F1-Score.
  - Confusion Matrix for class-wise performance.

**Evaluation Results**:

- **MLPClassifier Accuracy**: ~95%
- **LSTM Accuracy**: ~92%

---

### 5. **Model Deployment**

- Saved the trained **MLPClassifier** using `joblib`.
- Recreated preprocessing pipeline (scaler, label encoder) for new predictions.

**Saving the Model**:

```python
import joblib
joblib.dump(mlp, "mlp_classifier_model.joblib")
```

**Loading and Predicting**:

```python
# Load the model
loaded_model = joblib.load("mlp_classifier_model.joblib")

# Predict new data
predictions = loaded_model.predict(new_data_scaled)
```

---

## Results and Insights

- The **MLPClassifier** outperformed other models in terms of accuracy and simplicity.
- The project demonstrates how gait phase analysis can be approached using force data.
- Future improvements include:
  - Collecting more data for better generalization.
  - Exploring additional advanced models like `XGBoost`.

---

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/username/gait-phase-analysis.git
   cd gait-phase-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing and training notebook:

   ```bash
   jupyter notebook notebooks/preprocessing_and_training.ipynb
   ```

4. Use the trained model for predictions:
   ```bash
   jupyter notebook notebooks/prediction.ipynb
   ```

---

## Future Work

- Extend the project to predict actual gait phases (`Stance`, `Swing`, etc.).
- Incorporate additional sensor data (e.g., accelerometers).
- Deploy the model as an API using Flask or FastAPI.

---

## Acknowledgments

This project was developed by going through material in domain knowledge, articles (some of materials are linked below) and dataset and similar projects and also with guidance and context provided through interactions with ChatGPT, ensuring best practices in machine learning and deep learning.

external material for reference:
https://www.mad.tf.fau.de/research/activitynet/gaitphase-database/

material (papers, research, articles) for learning and getting inspired while doing this project:
https://drive.google.com/file/d/1gC5iiZM9-A_a0_9x29eDBw9qMCljv4Vy/view

https://my.clevelandclinic.org/health/diseases/21092-gait-disorders

https://pubmed.ncbi.nlm.nih.gov/15519595/

https://www.quora.com/Is-a-gait-analysis-worth-it-how-can-you-actually-change-the-way-you-walk

https://drive.google.com/file/d/1RJ61RvPTUiW51z1Ok4OJdbaEoWhr95ZF/view
