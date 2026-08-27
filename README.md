# 🏦 Banking Risk Decision Engine
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PySpark](https://img.shields.io/badge/PySpark-E25A1C.svg?style=flat&logo=Apache-Spark&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1.svg?style=flat&logo=PostgreSQL&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) 
![XGBoost](https://img.shields.io/badge/XGBoost-179C5C.svg?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=FastAPI&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E.svg?style=flat&logo=Amazon-AWS&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-green)

An end-to-end Machine Learning Operations (MLOps) pipeline designed to simulate a modern banking decision intelligence system. This project processes millions of financial transactions to engineer features, predicts Customer Lifetime Value (LTV), and detects zero-day fraud anomalies. The final models are containerized and deployed as a live REST API.

## 🧠 Model Architectures
This project utilizes a dual-model architecture to handle separate business domains:

### 1. Risk Anomaly Detection (PyTorch)
* **Type:** Autoencoder Neural Network (Unsupervised)
* **Input:** Massive transfer counts, average & max balance discrepancies
* **Layers:** 
    * Encoder: Dense (16) -> ReLU -> Dense (4) -> ReLU
    * Decoder: Dense (16) -> ReLU -> Dense (3)
* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE) - *High reconstruction error flags fraud.*

### 2. Commercial LTV Prediction (XGBoost)
* **Type:** Gradient Boosted Regressor (Supervised)
* **Input:** Average and maximum transaction sizes
* **Objective:** `reg:squarederror`
* **Performance:** Achieved baseline R² of 0.6520.

## 🛠️ Technologies Used
* **Python**: Core programming language.
* **PySpark**: Distributed big data processing and feature engineering.
* **PostgreSQL**: Relational data warehousing and advanced SQL feature aggregation.
* **PyTorch & XGBoost**: Machine learning model development.
* **FastAPI & Uvicorn**: High-performance REST API development.
* **Docker**: Application containerization for cloud deployment.
* **AWS EC2**: Public cloud hosting.

## 📂 Project Structure
```text
├── models/                     # Directory for saved model artifacts (.pth, .json, .pkl)
├── scripts/
│   ├── process_spark.py        # PySpark big data processing script
│   ├── build_warehouse.py      # PostgreSQL table generation & SQL queries
│   └── train_models.py         # Model training and evaluation script
├── app.py                      # The main FastAPI web application
├── Dockerfile                  # Container blueprint
├── requirements.txt            # List of Python dependencies
└── README.md                   # Project documentation

```

## 💻 How to Run Locally

1. **Clone the repository:**

```bash
git clone [https://github.com/sarthak-144/Banking-Risk-Decision-Engine.git](https://github.com/sarthak-144/Banking-Risk-Decision-Engine.git)
cd Banking-Risk-Decision-Engine

```

2. **Build the Docker container:**

```bash
docker build -t banking-risk-api .

```

3. **Run the application:**

```bash
docker run -d -p 8000:8000 banking-risk-api

```

4. **Test the API:**
Open your web browser and navigate to the interactive Swagger UI at:
`http://127.0.0.1:8000/docs`

## 📊 Features

* **Big Data Processing at Scale:** Processed over 6.3 million transaction rows using PySpark, demonstrating memory management and data partitioning.
* **Relational Data Marts:** Executed advanced SQL to aggregate granular transactions into highly optimized `account_risk_features` and `customer_ltv_features` tables.
* **Zero-Day Fraud Detection:** Deployed an unsupervised deep learning model capable of identifying novel fraud patterns that standard rules-based systems miss.
* **Containerized Deployment:** Packaged the inference environment into a lightweight, CPU-optimized Docker image.

## 📝 Dataset

The data pipeline is engineered around the **PaySim Dataset**, a synthetic dataset of mobile money transactions based on a sample of real financial logs from a mobile money service implemented in an African country.

* **Size:** 6,362,620 transaction records.
* **Timeline:** Simulates 30 days of transaction history (represented as 744 `steps`, where 1 step = 1 hour).
* **Core Features Used:** Transaction `type` (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER), `amount`, original account balances (`oldbalanceOrg`, `newbalanceOrig`), and destination account balances (`oldbalanceDest`, `newbalanceDest`).
* **Engineering:** The raw financial columns are transformed via PySpark into discrepancy metrics and massive-transfer flags to profile standard account behaviors for the Autoencoder and LTV models.

## 📜 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE)

```

```
