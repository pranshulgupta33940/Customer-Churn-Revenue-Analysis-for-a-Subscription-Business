-- 🔹 STEP 3 — SQL ANALYSIS
-- Assuming table name: customer_churn

-- 1. Churn Rate by Contract Type
SELECT 
    Contract,
    COUNT(customerID) as Total_Customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as Churned_Customers,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(customerID), 2) as Churn_Rate_Percent
FROM customer_churn
GROUP BY Contract
ORDER BY Churn_Rate_Percent DESC;

-- 2. Revenue Lost due to Churn
SELECT 
    SUM(MonthlyCharges) as Total_Monthly_Revenue_Lost
FROM customer_churn
WHERE Churn = 'Yes';

-- 3. Average Tenure of Churned vs Non-Churned Customers
SELECT 
    Churn,
    ROUND(AVG(tenure), 1) as Avg_Tenure_Months
FROM customer_churn
GROUP BY Churn;

-- 4. Top 5 Customer Segments by Churn Rate (Example segment: Contract + InternetService)
SELECT 
    Contract,
    InternetService,
    COUNT(customerID) as Total_Customers,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(customerID), 2) as Churn_Rate_Percent
FROM customer_churn
GROUP BY Contract, InternetService
HAVING COUNT(customerID) > 50 -- Filter for significant segments
ORDER BY Churn_Rate_Percent DESC
LIMIT 5;
