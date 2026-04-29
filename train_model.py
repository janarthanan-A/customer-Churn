import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train():
    # Load dataset
    data_path = 'data/customer_churn_dataset.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    # Features and Target
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Identify categorical and numerical columns
    categorical_cols = ['contract_type', 'internet_service', 'payment_method']
    numerical_cols = ['tenure', 'monthly_charges', 'total_charges', 'customer_support_calls']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }

    # Train and evaluate individual models
    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }

    # Create Ensemble Model (Voting Classifier)
    ensemble_clf = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    
    ensemble_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', ensemble_clf)])
    ensemble_pipeline.fit(X_train, y_train)
    y_pred_ensemble = ensemble_pipeline.predict(X_test)

    results['Ensemble (Voting)'] = {
        'Accuracy': accuracy_score(y_test, y_pred_ensemble),
        'Precision': precision_score(y_test, y_pred_ensemble),
        'Recall': recall_score(y_test, y_pred_ensemble),
        'F1 Score': f1_score(y_test, y_pred_ensemble)
    }

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save results
    joblib.dump(results, 'models/metrics.pkl')

    # Save the best model (Ensemble in this case as it usually performs best)
    joblib.dump(ensemble_pipeline, 'models/customer_churn_model.pkl')
    
    print("Training complete. Model and metrics saved.")
    return results

if __name__ == "__main__":
    train()
