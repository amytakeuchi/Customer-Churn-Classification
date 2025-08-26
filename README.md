# 📊 Telecom Churn Classification

**Goal: Predict customer churn for a telecom dataset using statistical analysis, feature engineering, and machine learning models.**

## Highlights

**Statistics & Math**
- Chi-square, ANOVA, correlation analysis for feature relevance.
- Normality checks (Q-Q plots, skew/kurtosis) to validate assumptions.
- Addressed multicollinearity and validated significance tests.

**Feature Engineering**
- Domain-inspired features (e.g., tenure buckets, payment type flags, service intensity).
- Ratio-based features (avg charges, service fees per product).
- Reduced noise via rigorous statistical testing.

**ML Pipeline**
- Train/Validation/Test split with stratification.
- SMOTE to correct churn imbalance (26.5% positive class).
- GridSearchCV hyperparameter tuning.
- Compared Logistic Regression, Random Forest, XGBoost, SVM, KNN, AdaBoost.

**Results**
- Best model: Logistic Regression (AUC = 0.84, F1 = 0.63, Accuracy = 0.81).
- Balances interpretability with performance.

## Exploratory Data Analysis (EDA) — Key Insights

**Data Cleaning**
- Found hidden nulls in TotalCharges (blank spaces not detected by .isnull()).
- Converted to float & imputed → ensured valid numerical analysis.

**Churn Distribution**
- 73% stayed vs 27% churned → clear imbalance, later addressed with SMOTE.

**Continuous Features**
- Churned customers: shorter tenure (~18 vs 38 months), lower total spend, but higher monthly charges.
- Non-normal distributions confirmed via KDE & Q-Q plots.

**Categorical Features**
- **Contract:** Month-to-month customers churn far more than 1–2 year contracts.
- **Payment Method:** Electronic check customers churn disproportionately.
- **Services:** Lack of Online Security, Backup, or Tech Support = higher churn.
- **Demographics:** Senior citizens, no partner, no dependents → higher churn risk.

**Interactions**
- High monthly charges raise churn risk across all contract types.
- Customers with bundled services churn less → service stickiness effect.

**Business Takeaways**
- Retention campaigns should target early-tenure, high-charge customers.
- Incentivize longer-term contracts.
- Promote value-added services (security, tech support) to reduce churn risk.

📔  *See full [EDA notebook](Cmain//telecom_churn_EDA.ipynb) for plots and detailed tests.*

## 📈 Modeling Workflow
- **Data Prep:** Cleaned types, imputed missing, dropped irrelevant features.
- **Feature Engineering:** Tenure buckets, service intensity, payment types, ratios.
- **Imbalance Handling:** Applied SMOTE to training data only.
- **Baseline Model:** Logistic Regression.
- **Model Comparison:** Tuned & tested Random Forest, XGBoost, SVM, KNN, AdaBoost.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1, ROC AUC.

I prioritized the Metrics in the following order:
- * F1 Score: Used as primary comparison metric between models, balancing precision and recall for optimal intervention efficiency
- * Recall (Sensitivity): Prioritized to maximize identification of potential churners, ensuring minimal missed opportunities for retention efforts
- * AUC-ROC: Selected as threshold-independent performance indicator to evaluate model discrimination ability across all possible classification thresholds
- * Accuracy: Considered but de-emphasized due to class imbalance in churn data (only ~27% customers churn)
Cross-Validation Approach

## 📊 Results
- Logistic Regression outperformed others with AUC 0.84.
- Feature importance analysis (RF, XGBoost, AdaBoost) confirmed:
- Tenure, contract type, and bundled services drive churn prediction.

## 🚀 Next Steps
- Deploy best model via FastAPI/Flask.
- Add drift monitoring & live churn predictions.
- Explore cost-sensitive learning to estimate financial impact of churn.
