import joblib
import pandas as pd
import numpy as np

def get_risk_category(probability):
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

def make_prediction(input_data):
    """
    input_data: dict containing feature values
    """
    try:
        model = joblib.load('models/customer_churn_model.pkl')
        
        # Convert input dict to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Ensure column order matches training data
        # 'tenure', 'monthly_charges', 'total_charges', 'contract_type', 'internet_service', 'customer_support_calls', 'payment_method'
        cols = ['tenure', 'monthly_charges', 'total_charges', 'contract_type', 'internet_service', 'customer_support_calls', 'payment_method']
        df_input = df_input[cols]
        
        # Predict probability
        prob = model.predict_proba(df_input)[0][1] # Probability of Churn (class 1)
        prediction = model.predict(df_input)[0]
        
        return {
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'probability': round(prob * 100, 2),
            'risk_category': get_risk_category(prob)
        }
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Test prediction
    test_input = {
        'tenure': 12,
        'monthly_charges': 70.5,
        'total_charges': 846.0,
        'contract_type': 'Month-to-month',
        'internet_service': 'Fiber optic',
        'customer_support_calls': 4,
        'payment_method': 'Electronic check'
    }
    print(make_prediction(test_input))
