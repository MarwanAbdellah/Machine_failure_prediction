# 🧠 Machine Failure Prediction using Sensor Data

This project leverages deep learning, Optuna hyperparameter tuning, and MLflow experiment tracking to build an effective classifier for predicting machine failures using sensor data.

---

## 📁 Dataset Description

The dataset used in this project is sourced from KaggleHub:

**Dataset URL:** `umerrtx/machine-failure-prediction-using-sensor-data`

It contains the following features:

| Feature       | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `footfall`    | Number of people or objects passing by the machine.                         |
| `tempMode`    | Temperature mode or setting of the machine.                                 |
| `AQ`          | Air quality index near the machine.                                         |
| `USS`         | Ultrasonic sensor readings (proximity).                                     |
| `CS`          | Current sensor readings (electrical current usage).                         |
| `VOC`         | Volatile organic compounds level.                                           |
| `RP`          | Rotational position or RPM.                                                 |
| `IP`          | Input pressure.                                                             |
| `Temperature` | Operating temperature of the machine.                                       |
| `fail`        | **Target**: 1 indicates failure, 0 indicates normal operation.              |

---

## 📊 Project Pipeline

### **1. Data Preprocessing**
- Resampling using **SMOTE** to handle class imbalance.
- Scaling using **StandardScaler**.
- Train-test split (80/20).

### **2. Model Architecture**
- Fully connected neural network built using **PyTorch**.
- Layers: Linear → BatchNorm → ReLU → Dropout (repeated).
- Final linear output layer for binary classification.

### **3. Hyperparameter Optimization**
- Performed using **Optuna**.
- Tuned Parameters:
  - Dropout rate
  - Learning rate
  - Number of hidden layers
  - Number of epochs

### **4. Training and Evaluation**
- Trained using `CrossEntropyLoss` and `Adam` optimizer.
- Training and validation metrics logged per epoch.
- Best model selected based on validation loss.

### **5. Experiment Tracking**
- All metrics, parameters, and models logged using **MLflow**.

---

## 🚀 How to Run

### **Requirements**
- Python ≥ 3.11
- PyTorch
- scikit-learn
- imbalanced-learn
- Optuna
- MLflow
- matplotlib / seaborn

### **Install Requirements**
```
pip install -r requirements.txt
```

### **Running the Notebook**
Use Jupyter or VS Code notebook interface:
```
jupyter notebook MachineFailurePrediction.ipynb
```
