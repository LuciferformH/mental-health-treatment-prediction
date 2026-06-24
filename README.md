# 🧠 Mental Health Treatment Prediction

A machine learning project that predicts whether a person in the tech industry is likely to seek mental health treatment based on workplace and personal factors.

## 📌 Overview

Mental health is a critical concern in the tech industry. This project uses machine learning to analyze survey data and predict whether an individual will seek mental health treatment. The goal is to help organizations understand the factors that influence treatment-seeking behavior and improve workplace mental health support.

## 📊 Dataset

- **Source**: [Mental Health in Tech Survey](https://www.kaggle.com/d/osmiamt/mental-health-in-tech-survey)
- **Size**: 1,259 responses
- **Features**: 26 features including demographics, work environment, benefits, and attitudes
- **Target**: `treatment` (Yes/No)

## 🎯 Problem Statement

Predict whether a person seeks mental health treatment using workplace and personal factors to help organizations improve their mental health support systems.

**Optimization Metric**: F1 Score

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, CatBoost
- **Visualization**: Matplotlib, Seaborn
- **Web App**: Streamlit
- **Model Serialization**: Joblib

## 📁 Project Architecture

```
mental-health-treatment-prediction/
│
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Processed data
│
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis
│
├── src/
│   ├── data_preprocessing.py   # Data cleaning & preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── train.py                # Model training
│   ├── evaluate.py             # Model evaluation
│   ├── predict.py              # Prediction logic
│   └── utils.py                # Utility functions
│
├── models/
│   └── best_model.pkl          # Saved best model
│
├── reports/
│   ├── confusion_matrix.png    # Confusion matrix plot
│   ├── feature_importance.png  # Feature importance plot
│   └── metrics.json            # Model metrics
│
├── app/
│   └── streamlit_app.py        # Streamlit web application
│
├── main.py                     # Main pipeline script
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mental-health-treatment-prediction.git
cd mental-health-treatment-prediction
```

2. **Create virtual environment** (optional but recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Place the dataset**
   - Download the dataset from Kaggle
   - Place `mental_health_survey.csv` in `data/raw/`

## 📖 Usage

### Run the Complete Pipeline
```bash
python main.py
```

### Run EDA Notebook
```bash
jupyter notebook notebooks/EDA.ipynb
```

### Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## 📈 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.7052 | 0.7227 | 0.6772 | 0.6992 |
| Decision Tree | 0.7530 | 0.7731 | 0.7244 | 0.7480 |
| Random Forest | 0.7450 | 0.7561 | 0.7323 | 0.7440 |
| XGBoost | 0.7371 | 0.7402 | 0.7402 | 0.7402 |
| LightGBM | 0.7490 | 0.7667 | 0.7244 | 0.7449 |
| **CatBoost** | **0.7689** | **0.7949** | **0.7323** | **0.7623** |

### Generated Reports
- Confusion Matrix
- ROC Curve
- Feature Importance Plot
- Classification Report

## 🔮 Future Improvements

1. **Advanced Feature Engineering**: Create more interaction features
2. **Deep Learning**: Experiment with neural networks
3. **Ensemble Methods**: Stack multiple models for better performance
4. **API Deployment**: Deploy model as REST API using FastAPI
5. **SHAP Analysis**: Add explainability with SHAP values
6. **Data Collection**: Gather more recent survey data
7. **Multi-class Classification**: Predict treatment urgency levels



## 🙏 Acknowledgments

- Dataset from Kaggle
- Open source ML libraries
- Mental health awareness community
