import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, recall_score
import joblib

# Load Data
df = pd.read_csv('cleaned_data.csv')

# --- Feature Engineering (Same as training) ---
def tenure_group(t):
    if t <= 12: return '0-12'
    elif t <= 24: return '12-24'
    else: return '24+'
df['tenure_group'] = df['tenure'].apply(tenure_group)

avg_charges = df['MonthlyCharges'].mean()
df['high_value_customer'] = df['MonthlyCharges'].apply(lambda x: 1 if x > avg_charges else 0)

df['payment_risk_flag'] = df.apply(lambda x: 1 if x['PaymentMethod'] == 'Electronic check' and x['Contract'] == 'Month-to-month' else 0, axis=1)

# --- Encoding ---
le = LabelEncoder()
target = 'Churn'
data_ml = df.drop('customerID', axis=1)

cat_cols = data_ml.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != target:
        data_ml[col] = le.fit_transform(data_ml[col])
    else:
        # Check unique values to ensure mapping is correct
        # Depending on generation, it might be Yes/No
        data_ml[col] = data_ml[col].map({'Yes': 1, 'No': 0})

X = data_ml.drop(target, axis=1)
y = data_ml[target]

# Split (Same random_state as training to get valid test metrics)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load Model
model = joblib.load('model.pkl')

# Predict
y_pred = model.predict(X_test)

# Metrics
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

r = recall_score(y_test, y_pred)
print(f"Recall: {r:.4f}")
