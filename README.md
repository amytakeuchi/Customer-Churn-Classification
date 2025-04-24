# Customer Churn Classification Project
## Project Overview
This project aims to predict customer churn using various classification models. The end-to-end pipeline includes data preprocessing, exploratory data analysis (EDA), feature engineering, class imbalance handling, model training, hyperparameter tuning, and evaluation.

## 🛠️ Setup & Library Installation
Installed essential libraries for:
- Data manipulation: pandas, numpy
- Visualization: matplotlib, seaborn
- Modeling: scikit-learn, xgboost, imbalanced-learn
- Utilities: math, warnings, joblib

## Data Preprocessing
- Loaded and cleaned raw data
- Handled missing values
- Converted data types (e.g., TotalCharges to float)
- Cleaned categorical strings (e.g., stripped whitespaces, unified values)

## 📊 Exploratory Data Analysis (EDA)
- Analyzed class distribution of the target variable Churn
- Explored relationships between churn and features using:
  - Count plots
  - Histograms
  - Density plots
  - Boxplots
  - Investigated correlation among numerical features

## Feature Assessment
- Identified and separated categorical and numerical features
- Checked distributions and cardinality of categorical features
- Evaluated numerical features for skewness and scaling needs
- Performed multicollinearity checks using Variance Inflation Factor (VIF)

## 🧠 Feature Engineering
- Created meaningful derived features such as:
  - new_num_of_services
  - new_automatic_payment
  - new_high_monthly_charges_with_engaged
  - Tenure-based categories like new_tenure_year_0-1 Year
- Removed multicollinear and redundant features after VIF analysis

## 📂 Train/Test/Validation Split
- Performed stratified splitting to maintain churn ratio in all sets
- Split data into:
  - Training set
  - Validation set
  - Test set

## Handling Class Imbalance
- Applied SMOTE oversampling on training data to balance churn and non-churn classes

## Modeling and Baseline
Trained a baseline Logistic Regression model without tuning for reference

## Hyperparameter Tuning & Model Comparison
- Used GridSearchCV with cross-validation to train and tune:
  - ✅ Logistic Regression (liblinear, C, penalty)
  - 🌲 Random Forest (n_estimators, max_depth, min_samples_split, min_samples_leaf)
  - ⚡ XGBoost (max_depth, learning_rate, subsample, colsample_bytree)
  - 🎯 AdaBoost (n_estimators, learning_rate)
### Model Selection Rationale

For this customer churn prediction project, I strategically selected four complementary classification algorithms, each chosen for specific strengths:

#### Logistic Regression
- Serves as an interpretable baseline with coefficients directly showing how each feature influences churn probability
- Provides well-calibrated probability scores for flexible threshold adjustment based on business needs (e.g., prioritizing retention of high-value customers)
- Offers regularization options to prevent overfitting when working with numerous customer features
- Enables straightforward communication of churn drivers to business stakeholders

#### Random Forest
- Captures complex non-linear relationships between customer attributes that simple models might miss
- Performs well without feature scaling, reducing preprocessing requirements for production deployment
- Shows resilience to outliers in customer behavior data
- Provides built-in feature importance measures to identify key churn indicators
- Handles interaction effects between features (e.g., how contract type and tenure jointly affect churn)

#### XGBoost
- Delivers state-of-the-art performance on structured customer data through gradient boosting
- Efficiently handles missing values common in customer datasets without requiring imputation
- Implements built-in regularization to prevent overfitting on training data
- Excels at identifying subtle patterns in customer behavior leading to churn
- Optimizes computational resources through parallelization for faster training and iteration

#### AdaBoost
- Focuses sequentially on hard-to-classify customer segments that other models might miss
- Addresses class imbalance through its weighted sample approach (important for churn datasets where churners are typically the minority class)
- Provides complementary insights to other ensemble methods by identifying different patterns
- Requires fewer hyperparameters than XGBoost, making it easier to tune while maintaining strong performance

This combination of models creates a robust approach that balances interpretability, predictive power, and practical implementation considerations - allowing us to not only predict which customers might churn but also understand why they're likely to do so.

## 📈 Model Evaluation
Evaluated each model using:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC (ROC Curve)
- Confusion Matrix

<br/>
  As a result of model evaluation, Logistic Regression model was identified as the most efffective model for Churn prediction for this project.
<br/>
## Feature Importance Analysis
- Extracted feature importances from:
  - Random Forest (Gini importance)
  - XGBoost (gain-based importance)
  - AdaBoost
- Interpreted the most influential features in churn prediction



