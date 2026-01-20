import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading data...")
df = pd.read_csv('cleaned_data.csv')

# --- Feature Engineering (Same as before) ---
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

# Dictionary to store encoders for app usage if needed, 
# but for simplicity we'll just encode and save the model.
# In a real app, we'd need to save the encoders too. 
# We'll handle encoding in the app manually or use a pipeline. 
# For this script we assume input to app will be handled.

cat_cols = data_ml.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != target:
        data_ml[col] = le.fit_transform(data_ml[col])
    else:
        data_ml[col] = data_ml[col].map({'Yes': 1, 'No': 0})

X = data_ml.drop(target, axis=1)
y = data_ml[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Hyperparameter Tuning ---
print("Starting Grid Search...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

y_pred = best_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Optimized Accuracy: {acc:.4f}")

# --- Save Model ---
joblib.dump(best_rf, 'model.pkl')
print("Model saved to model.pkl")

# Save feature names to ensure order in app
joblib.dump(X.columns.tolist(), 'features.pkl')
print("Feature names saved to features.pkl")
