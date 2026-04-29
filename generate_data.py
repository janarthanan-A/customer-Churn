import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

# Number of records
n = 1000

# Generating features
tenure = np.random.randint(1, 72, n)
monthly_charges = np.random.uniform(20, 120, n)
total_charges = tenure * monthly_charges + np.random.normal(0, 50, n)
total_charges = np.maximum(total_charges, monthly_charges) # Ensure total >= monthly

contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n)
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n)
customer_support_calls = np.random.randint(0, 9, n)
payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n)

# Logic for churn (higher risk for month-to-month, fiber optic, high support calls, low tenure)
churn_prob = (
    (contract_type == 'Month-to-month') * 0.4 +
    (internet_service == 'Fiber optic') * 0.2 +
    (customer_support_calls > 3) * 0.3 +
    (tenure < 12) * 0.2 +
    np.random.normal(0, 0.1, n)
)

churn = (churn_prob > 0.6).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'contract_type': contract_type,
    'internet_service': internet_service,
    'customer_support_calls': customer_support_calls,
    'payment_method': payment_method,
    'churn': churn
})

# Save to CSV
os.makedirs('data', exist_ok=True)
df.to_csv('data/customer_churn_dataset.csv', index=False)

print("Dataset generated successfully.")
