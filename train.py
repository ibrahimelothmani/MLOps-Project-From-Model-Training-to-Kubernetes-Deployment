import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Feature selection
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model persistence
joblib.dump(model, "diabetes_model.pkl")