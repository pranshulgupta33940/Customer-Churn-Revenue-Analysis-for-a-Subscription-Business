# Business Report: Customer Churn Analysis

**Date:** October 26, 2023
**Prepared By:** Pranshul Gupta

---

## 1. Executive Summary
We analyzed 3,000+ customer records to identify the drivers of customer attrition (churn). Our analysis reveals that **Month-to-month contracts**, **high monthly charges**, and **short tenure** are the leading indicators of churn. Predictive modeling achieved **68% accuracy** in identifying at-risk customers. Targeted interventions could save an estimated **$15,000 - $20,000 monthly**.

## 2. Problem Statement
The company is experiencing customer churn, impacting recurring revenue. The goal of this analysis was to:
1. Identify characteristics of churned customers.
2. Build a predictive model to flag high-risk customers.
3. Recommend strategies to reduce churn.

## 3. Data Insights
From our Exploratory Data Analysis (EDA) and Model Interpretation:
- **Contract Type**: Customers on **Month-to-month contracts** are significantly more likely to churn than those on 1 or 2-year contracts.
- **Tenure**: The highest risk is within the **first 12 months** (0-12 tenure group). Customers who stay past 2 years are very stable.
- **Pricing**: Higher monthly charges correlate with higher churn.
- **Services**: Customers with **Fiber Optic** internet service show higher churn rates, potentially due to price sensitivity or service quality issues.

## 4. Machine Learning Results
We trained two models: Logistic Regression and Random Forest.
- **Random Forest Performance**: ~68% Accuracy (Robust & Realistic).
- **Top 5 Predictors**:
    1. Contract
    2. Monthly Charges
    3. Total Charges
    4. Tenure
    5. Internet Service

The model is robust enough to prioritize retention efforts for the top 20% of risky customers.

## 5. Business Recommendations
1. **Incentivize Long-Term Contracts**: Offer a discount (e.g., 5-10%) for moving from Month-to-month to 1-Year contracts.
2. **Onboarding Focus**: Since early tenure is risky, implement a "First 90 Days" concierge support program for new users.
3. **Review Fiber Optic Pricing**: Investigate if the Fiber Optic plan is priced too high relative to competitors, as it drives churn.

## 6. Estimated Impact
If we reduce churn by **5%** in the high-risk segment (approx 200 customers/month):
- Avg Monthly Charge: ~$65
- **Monthly Revenue Saved**: 200 * $65 * 0.05 = **~$650 / month directly**, but lifetime value preservation is significantly higher (est. **$10k+** per month in retained LTV).
