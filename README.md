# MLOps-Project-From-Model-Training-to-Kubernetes-Deployment

📋 Project Overview
This project implements a complete MLOps pipeline for a diabetes prediction model using:

Machine Learning: Random Forest Classifier on the Pima Indians Diabetes Dataset
API Framework: FastAPI for creating REST endpoints
Containerization: Docker for packaging the application
Orchestration: Kubernetes for deployment and scaling
The model predicts diabetes risk based on five key health metrics: Pregnancies, Glucose levels, Blood Pressure, BMI, and Age.


# Key Learnings:

✅ Reproducibility: Using a fixed random_state=42 ensures consistent results across runs

✅ Model Serialization: joblib is efficient for saving scikit-learn models, preserving the entire model state

✅ Feature Engineering: Selected only relevant features to keep the model simple and interpretable

✅ Data Source Management: Using a hosted dataset URL makes the project portable and easy to reproduce