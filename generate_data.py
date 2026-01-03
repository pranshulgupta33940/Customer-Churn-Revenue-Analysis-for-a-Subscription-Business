import pandas as pd
import numpy as np
import random

def generate_churn_data(n_samples=3000):
    np.random.seed(42)
    random.seed(42)
    
    data = {
        'customerID': [f'{random.randint(1000,9999)}-{random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")*4}' for _ in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples).round(2),
        'Churn': np.random.choice(['Yes', 'No'], n_samples) # Base churn, will adjust for correlations later
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some realistic dirt/noise
    # 1. TotalCharges should be Tenure * MonthlyCharges roughly, but let's make it a string and have some missing values
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
    df['TotalCharges'] = df['TotalCharges'].apply(lambda x: " " if x == 0 else str(round(x + np.random.normal(0, 10), 2)))
    
    # Introduce duplicate rows
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    
    # Introduce missing values in Churn (make it harder)
    # Actually, target shouldn't have missing usually, let's put missing in TotalCharges
    df.loc[random.sample(range(len(df)), 20), 'TotalCharges'] = np.nan
    
    # Adjust Churn correlation so models actually work
    # Higher tenure -> Lower Churn
    # Month-to-month -> Higher Churn
    # Fiber optic -> Higher Churn
    
    # Realistic Churn Logic for Resume (Target 75-80% Accuracy)
    # This mimics real-world noise where behavior is not perfectly predictable.
    
    def adjust_churn(row):
        score = 0
        # Moderate Drivers
        if row['tenure'] < 12: score += 2
        if row['Contract'] == 'Month-to-month': score += 2
        if row['InternetService'] == 'Fiber optic': score += 1
        if row['TotalCharges'] == " ": score -= 1 # New customers
        
        # Probabilistic approach (Sigmoid)
        # Score 0 -> Prob 0.5 (random)
        # Score +3 -> Prob ~0.95
        
        prob = 1 / (1 + np.exp(-(score - 2))) # Bias slightly to No Churn to create imbalance
        
        return 'Yes' if random.random() < prob else 'No'

    df['Churn'] = df.apply(adjust_churn, axis=1)
    
    print(f"Generated dataset with {len(df)} rows.")
    return df

if __name__ == "__main__":
    df = generate_churn_data()
    df.to_csv('churn_dataset.csv', index=False)
    print("Saved to churn_dataset.csv")
