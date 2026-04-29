# Customer Churn Prediction using Ensemble Machine Learning

This project provides a complete end-to-end solution for predicting customer churn using a high-performance Ensemble model (Voting Classifier).

## Features
- **Ensemble AI**: Combines Random Forest, XGBoost, and Gradient Boosting for superior accuracy.
- **Modern UI**: Built with Bootstrap, AOS (Animate on Scroll), and Chart.js.
- **Real-time Prediction**: Detailed risk analysis with churn probability.
- **Full Stack**: Flask backend with RESTful API support.
- **CI/CD Ready**: GitHub Actions pipeline and Render deployment config included.

## Project Structure
```
customer-churn-prediction/
├── app.py                  # Flask Backend
├── train_model.py          # Model Training & Ensemble Logic
├── predict.py              # Prediction Helper
├── requirements.txt        # Dependencies
├── render.yaml             # Render Deployment Config
├── data/
│   └── customer_churn_dataset.csv
├── models/                 # Generated after training
│   ├── customer_churn_model.pkl
│   └── metrics.pkl
├── templates/
│   ├── index.html          # Landing Page
│   └── predict.html        # Prediction Form
└── .github/
    └── workflows/
        └── pipeline.yml    # CI/CD Workflow
```

## How to Run Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**:
   ```bash
   python train_model.py
   ```

3. **Start the Flask App**:
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

## Deployment
This project is pre-configured for **Render**. Simply connect your GitHub repository and Render will use `render.yaml` to build and deploy the application.
