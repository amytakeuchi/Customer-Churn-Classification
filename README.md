# Customer-Churn-Classification
This project develops classification algorithms to predict Telco Customer Churn 


# TOC:
This project follows the following Table of Contents:
- 1. Project Setup & Data Preparation
  2. Exploratory Data Analysis (EDA)
  3. Initial Data Preprocessing
  4. Data Splitting
  5. Advanced Preprocessing (Training Data Only)
  6. Feature Engineering & Selection (Training Data Only)
  7. Model Building & Evaluation
  8. Model Optimization
  9. Test Set Evaluation
  10. Deployment Preparation

# 1. Project Setup & Data Preparation

- Dataset introduction & business context
- Import libraries and load data
- Quick data summary (shape, types, missing values)

# 2. Exploratory Data Analysis (EDA)

- Basic statistics and distribution analysis
- Check distribution of continuous variables
- Histograms, density plots, Q-Q plots
- Test for normality (Shapiro-Wilk, etc.)
- Identify skewness and kurtosis
- Target variable analysis (churn rate)
- Key feature visualization (focus only on likely predictors)
- Linearity check between features and target variable
- Collinearity analysis:
  - Correlation matrix and heatmap for numerical features
  - Chi-square test of independence for categorical features
  - Cramer's V coefficient for categorical associations
  - Point-biserial correlation for numerical-categorical relationships


# 3. Initial Data Preprocessing

- Handle missing values
- Outlier detection and treatment
- Feature encoding (categorical to numerical)
    - Convert categorical variables for further analysis

# 4. Data Splitting

- Split data into train-test-validation sets
- Ensure proper stratification for the target variable

# 5. Advanced Preprocessing (Training Data Only)

- Class imbalance assessment
- Oversampling/undersampling techniques for handling imbalance
- Feature scaling

# 6. Feature Engineering & Selection (Training Data Only)

- Create interaction features if needed
- Multicollinearity assessment:
    - Variance Inflation Factor (VIF) analysis
    - Addressing highly collinear features
- Feature selection methods:
    - Chi-square tests for categorical variables
    - ANOVA for continuous variables with categorical target
    - Correlation analysis for continuous variables
- Feature importance ranking (Random Forest)
- Dimensionality reduction if necessary (PCA)
- Select optimal feature subset

# 7. Model Building & Evaluation

- Baseline model implementation
- Model comparison (Logistic Regression, Random Forest, XGBoost, etc.)
- Cross-validation of best performers
- Apply same transformations to validation data

# 8. Model Optimization

- Hyperparameter tuning of best model
- Threshold optimization
- Final model evaluation (confusion matrix, ROC-AUC, precision-recall)

# 9. Test Set Evaluation

- Apply all preprocessing and transformations to test set
- Evaluate final model performance on unseen data

# 10. Deployment Preparation

- Model serialization (saving)
- Interpretation of results & business insights
- Summary of findings & recommendations
