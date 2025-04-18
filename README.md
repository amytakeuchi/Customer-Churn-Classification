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



