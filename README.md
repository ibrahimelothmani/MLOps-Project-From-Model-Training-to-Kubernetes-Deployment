# 🚀 MLOps Project: From Model Training to Kubernetes Deployment

A complete end-to-end MLOps pipeline demonstrating how to train a machine learning model, containerize it with Docker, and deploy it to Kubernetes with a production-ready API.

## 📋 Project Overview

This project implements a **diabetes prediction model** deployed as a REST API using modern MLOps practices. The model predicts diabetes risk based on five key health metrics: Pregnancies, Glucose levels, Blood Pressure, BMI, and Age.

### Technology Stack

- **Machine Learning**: scikit-learn (Random Forest Classifier)
- **Dataset**: [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)
- **API Framework**: FastAPI
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Model Serialization**: joblib

---

## 🏗️ Architecture & Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  train.py       │────▶│  Model Training  │────▶│  model.pkl      │────▶│  main.py         │
│  (Data + Train) │     │  Random Forest   │     │  (Serialized)   │     │  (FastAPI)       │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
                                                                                    │
                                                                                    ▼
                                                                          ┌──────────────────┐
                                                                          │  Dockerfile      │
                                                                          │  (Containerize)  │
                                                                          └──────────────────┘
                                                                                    │
                                                                                    ▼
                                                                          ┌──────────────────┐
                                                                          │  Docker Image    │
                                                                          │  (Push to Hub)   │
                                                                          └──────────────────┘
                                                                                    │
                                                                                    ▼
                                                                          ┌──────────────────┐
                                                                          │  k8s-deploy.yml  │
                                                                          │  (Kubernetes)    │
                                                                          └──────────────────┘
```

---

## 📁 Project Structure

```
MLOps-Project-From-Model-Training-to-Kubernetes-Deployment/
│
├── train.py              # Model training script
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── k8s-deploy.yml        # Kubernetes deployment manifest
├── diabetes_model.pkl    # Serialized trained model
└── README.md             # Project documentation
```

---

## 🔍 What Happens Under the Hood

### 1️⃣ **Model Training (`train.py`)**

#### What it does:
- Downloads the Pima Indians Diabetes dataset from a remote URL
- Selects 5 relevant features for prediction
- Splits data into training (80%) and testing (20%) sets
- Trains a Random Forest Classifier
- Serializes the model using joblib

#### Key Code Flow:
```python
# Load data from remote source
df = pd.read_csv(url)

# Feature selection - only 5 most important features
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Train-test split with fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model to disk
joblib.dump(model, "diabetes_model.pkl")
```

#### Why Random Forest?
- Handles non-linear relationships well
- Resistant to overfitting
- Provides feature importance insights
- No extensive preprocessing required

---

### 2️⃣ **API Development (`main.py`)**

#### What it does:
- Creates a FastAPI application with two endpoints
- Loads the pre-trained model from disk
- Validates input data using Pydantic models
- Returns predictions in JSON format

#### Endpoints:

**`GET /`** - Health check
```json
{
  "message": "Diabetes Prediction API is live"
}
```

**`POST /predict`** - Make predictions
```json
// Request
{
  "Pregnancies": 2,
  "Glucose": 120.0,
  "BloodPressure": 70.0,
  "BMI": 25.5,
  "Age": 30
}

// Response
{
  "diabetic": false
}
```

#### Key Features:
- **Pydantic Validation**: Ensures input data types are correct
- **Model Loading**: Loads `diabetes_model.pkl` at startup (not on every request)
- **NumPy Array Conversion**: Converts input to the format expected by scikit-learn
- **Boolean Output**: Returns a clear true/false prediction

---

### 3️⃣ **Containerization (`Dockerfile`)**

#### What it does:
- Creates a lightweight, reproducible container image
- Installs all Python dependencies
- Copies application code and trained model
- Exposes port 8000 for the API

#### Layer-by-Layer Breakdown:
```dockerfile
FROM python:3.10                    # Base image with Python 3.10
WORKDIR /app                        # Set working directory inside container
COPY . /app                         # Copy all project files to container
RUN pip install -r requirements.txt # Install dependencies
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Why Docker?
- **Consistency**: Same environment everywhere (dev, staging, prod)
- **Isolation**: Dependencies don't conflict with host system
- **Portability**: Runs on any machine with Docker installed
- **Versioning**: Images are tagged and versioned

---

### 4️⃣ **Kubernetes Deployment (`k8s-deploy.yml`)**

#### What it does:
- Defines a **Deployment** with 2 replicas for high availability
- Creates a **LoadBalancer Service** to expose the API externally
- Ensures automatic restarts if containers fail
- Enables horizontal scaling

#### Components Explained:

**Deployment:**
```yaml
spec:
  replicas: 2                    # Run 2 pods for redundancy
  selector:
    matchLabels:
      app: diabetes-api          # Match pods with this label
  template:
    spec:
      containers:
      - name: diabetes-api
        image: aman202004/diabetes-api:latest  # Docker Hub image
        ports:
        - containerPort: 8000    # Container listens on port 8000
        imagePullPolicy: Always  # Always pull latest image
```

**Service (LoadBalancer):**
```yaml
spec:
  type: LoadBalancer    # Cloud provider will provision an external IP
  ports:
  - protocol: TCP
    port: 80            # External port (standard HTTP)
    targetPort: 8000    # Forward to container port 8000
```

#### How Kubernetes Works Here:
1. **Deployment controller** ensures 2 pods are always running
2. **Pods** contain the containerized FastAPI application
3. **Service** distributes incoming traffic across both pods
4. **LoadBalancer** provides a single external IP address
5. If a pod crashes, Kubernetes automatically replaces it

---

## 🚦 Getting Started

### Prerequisites
- Python 3.10+
- Docker
- Kubernetes cluster (Minikube, EKS, GKE, AKS, etc.)
- kubectl configured

### Step 1: Train the Model
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (creates diabetes_model.pkl)
python train.py
```

### Step 2: Test the API Locally
```bash
# Run the FastAPI application
uvicorn main:app --reload

# Test in another terminal
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "BMI": 25.5,
    "Age": 30
  }'
```

### Step 3: Build and Push Docker Image
```bash
# Build the image
docker build -t your-dockerhub-username/diabetes-api:latest .

# Test locally
docker run -p 8000:8000 your-dockerhub-username/diabetes-api:latest

# Push to Docker Hub
docker push your-dockerhub-username/diabetes-api:latest
```

### Step 4: Deploy to Kubernetes
```bash
# Update k8s-deploy.yml with your Docker Hub username
# Then apply the deployment
kubectl apply -f k8s-deploy.yml

# Check deployment status
kubectl get pods
kubectl get services

# Get the external IP (may take a few minutes)
kubectl get service diabetes-api-service
```

### Step 5: Test the Deployed API
```bash
# Replace EXTERNAL-IP with the LoadBalancer IP
curl -X POST "http://EXTERNAL-IP/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148.0,
    "BloodPressure": 72.0,
    "BMI": 33.6,
    "Age": 50
  }'
```

---

## 🎓 Key MLOps Concepts Demonstrated

### ✅ **Reproducibility**
- Fixed `random_state=42` ensures consistent model training results
- `requirements.txt` pins all dependencies
- Docker ensures same environment everywhere

### ✅ **Model Serialization**
- `joblib` efficiently saves scikit-learn models
- Model is loaded once at startup, not on every request
- Single source of truth for the trained model

### ✅ **API Design**
- RESTful API with clear POST endpoint for predictions
- Input validation using Pydantic models
- JSON input/output for easy integration

### ✅ **Containerization**
- Application + dependencies + model bundled together
- Lightweight Python 3.10 base image
- Single command to run the entire application

### ✅ **Orchestration**
- High availability with 2 replicas
- Automatic load balancing
- Self-healing (pods restart on failure)
- Easy horizontal scaling (`kubectl scale`)

### ✅ **Production Best Practices**
- Health check endpoint (`GET /`)
- Graceful error handling
- Structured logging via uvicorn
- External IP for public access

---

## 🔧 Useful Commands

### Docker
```bash
# Build image
docker build -t diabetes-api .

# Run container
docker run -d -p 8000:8000 diabetes-api

# View logs
docker logs <container-id>

# Stop container
docker stop <container-id>
```

### Kubernetes
```bash
# Get all resources
kubectl get all

# View pod logs
kubectl logs <pod-name>

# Scale deployment
kubectl scale deployment diabetes-api --replicas=3

# Delete deployment
kubectl delete -f k8s-deploy.yml

# Access pod shell
kubectl exec -it <pod-name> -- /bin/bash
```

---

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 5 (Pregnancies, Glucose, BloodPressure, BMI, Age)
- **Target**: Outcome (0 = No Diabetes, 1 = Diabetes)
- **Dataset**: 768 samples from Pima Indians Diabetes Database
- **Train/Test Split**: 80/20

---

## 🔮 Future Enhancements

- [ ] Add model versioning and A/B testing
- [ ] Implement CI/CD pipeline (GitHub Actions)
- [ ] Add monitoring and logging (Prometheus, Grafana)
- [ ] Include model performance metrics endpoint
- [ ] Add authentication and rate limiting
- [ ] Implement model retraining pipeline
- [ ] Add comprehensive unit and integration tests

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)