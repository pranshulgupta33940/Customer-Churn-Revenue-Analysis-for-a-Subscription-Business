import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import os

# Set style
sns.set(style="whitegrid")
if not os.path.exists('plots'):
    os.makedirs('plots')

print("🔹 STEP 1 — DATA CLEANING")
# Load Data
df = pd.read_csv('churn_dataset.csv')
print(f"Initial Shape: {df.shape}")

# Handle Missing/Wrong types
# TotalCharges is a string (e.g., " "), convert to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing values (TotalCharges has some NaN now)
print(f"Missing values before drop:\n{df.isnull().sum()}")
df.dropna(inplace=True)

# Remove duplicates
print(f"Duplicates before drop: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# Save cleaned
df.to_csv('cleaned_data.csv', index=False)
print("Saved cleaned_data.csv")
print(df.describe())

print("\n🔹 STEP 2 — EDA")
# Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.savefig('plots/churn_distribution.png')
print("Saved plots/churn_distribution.png")

# Churn vs Contract
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.savefig('plots/churn_by_contract.png')
print("Saved plots/churn_by_contract.png")

# Churn vs Tenure (Boxplot)
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure Distribution by Churn')
plt.savefig('plots/churn_by_tenure.png')
print("Saved plots/churn_by_tenure.png")

# Churn vs MonthlyCharges (KDE)
plt.figure(figsize=(8,5))
sns.kdeplot(df[df['Churn']=='Yes']['MonthlyCharges'], shade=True, color='red', label='Churn: Yes')
sns.kdeplot(df[df['Churn']=='No']['MonthlyCharges'], shade=True, color='blue', label='Churn: No')
plt.title('Monthly Charges Distribution by Churn')
plt.legend()
plt.savefig('plots/churn_by_monthly_charges.png')
print("Saved plots/churn_by_monthly_charges.png")

# Correlations
# Encode Churn for correlation
df_corr = df.copy()
df_corr['Churn'] = df_corr['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
numeric_df = df_corr.select_dtypes(include=[np.number])
plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('plots/correlation_matrix.png')
print("Saved plots/correlation_matrix.png")

print("\n🔹 STEP 4 — FEATURE ENGINEERING")
# Tenure Group
def tenure_group(t):
    if t <= 12: return '0-12'
    elif t <= 24: return '12-24'
    else: return '24+'
df['tenure_group'] = df['tenure'].apply(tenure_group)

# High Value Customer
avg_charges = df['MonthlyCharges'].mean()
df['high_value_customer'] = df['MonthlyCharges'].apply(lambda x: 1 if x > avg_charges else 0)

# Payment Risk Flag (e.g., Electronic check + Month-to-month often risky)
df['payment_risk_flag'] = df.apply(lambda x: 1 if x['PaymentMethod'] == 'Electronic check' and x['Contract'] == 'Month-to-month' else 0, axis=1)

print("Features created: tenure_group, high_value_customer, payment_risk_flag")

print("\n🔹 STEP 5 — MACHINE LEARNING")
# Encoder
le = LabelEncoder()
target = 'Churn'

# Drop customerID
data_ml = df.drop('customerID', axis=1)

# Categorical columns
cat_cols = data_ml.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != target:
        data_ml[col] = le.fit_transform(data_ml[col])
    else:
        # Manually map for clarity
        data_ml[col] = data_ml[col].map({'Yes': 1, 'No': 0})

X = data_ml.drop(target, axis=1)
y = data_ml[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:,1]))
print(classification_report(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train) # Tree models don't strictly need scaling, but it's fine
y_pred_rf = rf.predict(X_test)

print("--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
print(classification_report(y_test, y_pred_rf))

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

print("\nTop 5 Churn Drivers (Random Forest):")
for i in range(5):
    print(f"{i+1}. {features[indices[i]]} ({importances[indices[i]]:.4f})")

print("\nAnalysis Complete.")
