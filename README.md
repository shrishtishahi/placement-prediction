# Student Placement Prediction System

This project predicts whether a student will get placed in campus placements based on their academic and skill details. It uses an Artificial Neural Network (ANN) for prediction.

## How to Run

1. Install required libraries:
```
pip install -r requirements.txt
```

2. Run the app:
```
streamlit run app.py
```

3. Open http://localhost:8501 in your browser

## Files in this project

- app.py - Streamlit web app (Home + Prediction pages)
- Student_Placement_Prediction.ipynb - Jupyter notebook with step by step implementation
- Placement_Prediction_data.csv - Dataset of 10,000 students
- placement_ann_model.zip - Trained ANN model
- scaler.pkl - StandardScaler for feature scaling
- label_encoder.pkl - LabelEncoder for target variable
- requirements.txt - List of libraries needed

## Project Steps

1. Data Collection - 10,000 student records
2. Data Cleaning - removed IDs and duplicates
3. EDA - analyzed patterns and correlations
4. Data Preprocessing - encoding, scaling, train-test split
5. Model Building - ANN with 2 hidden layers
6. Model Evaluation - 93.67% accuracy
7. Deployment - Streamlit web application

## Model Architecture

Input (11 features) → Dense(64, ReLU) → Dropout(30%) → Dense(32, ReLU) → Dropout(30%) → Output(1, Sigmoid)

- Optimizer: Adam
- Loss: Binary Crossentropy
- Parameters: 2,881
- Test Accuracy: 93.67%
