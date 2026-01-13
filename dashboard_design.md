# Dashboard Design: Churn Analysis (Power BI)

## 1. Overview
A strategic dashboard for management to monitor customer attrition, identify at-risk segments, and track revenue impact.

## 2. KPI Cards (Top Row)
- **Overall Churn Rate**: displayed as a %, with a target comparison (e.g., vs last month or industry avg of 20%).
- **Total Customers**: Current active + churned count.
- **Lost Revenue (Monthly)**: Sum of Monthly Charges for Churned customers.
- **Revenue at Risk**: Predicted High Risk customers * Monthly Charges.

## 3. Visualizations
### A. Churn Overview
- **Chart**: Donut Chart
- **Dimensions**: Churn (Yes/No)
- **Insight**: Quick visual of the split (e.g., 26.5% Churn).

### B. Churn by Contract Type
- **Chart**: Clustered Bar Chart
- **X-Axis**: Contract (Month-to-month, One year, Two year)
- **Y-Axis**: Churn Rate %
- **Insight**: Highlight that Month-to-month users have the highest churn (expected ~40%+).

### C. Churn vs Tenure Group
- **Chart**: 100% Stacked Column Chart
- **X-Axis**: Tenure Group (0-12, 12-24, 24+)
- **Y-Axis**: % Churn
- **Legend**: Churn (Yes/No)
- **Insight**: Show early-tenure customers are most volatile.

### D. Monthly Churn Trend
- **Chart**: Line Chart
- **X-Axis**: Tenure (Months) - *Proxy for time in this dataset*
- **Y-Axis**: Count of Churned Customers
- **Insight**: Spike in churn at months 1-3.

### E. Demographics & Services (Slicer/Toggle)
- **Chart**: Matrix / Heatmap
- **Rows**: Internet Service / Payment Method
- **Values**: Churn Rate
- **Insight**: Fiber Optic + Electronic Check users often show high churn.

## 4. Filters / Slicers (Side Panel)
- **Contract Type**
- **Payment Method**
- **Senior Citizen (Yes/No)**
- **Tenure Group**

## 5. Color Palette
- **Primary**: Deep Blue (Safe), Red (Churn/Danger).
- **Secondary**: Light Grays for background.
