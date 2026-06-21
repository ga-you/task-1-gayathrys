# Data Science Project 1 — Advanced EDA & Feature Engineering

Cleaned and engineered the Titanic dataset using statistical imputation, IQR-based outlier handling, feature engineering, one-hot encoding, multicollinearity removal, and Pandera schema validation.

## Steps performed
- Handled missing values (median imputation for Age, dropped Cabin, dropped rows missing Embarked)
- Capped outliers in Age and Fare using the IQR method
- Engineered new features: family_size, is_alone, fare_per_person
- One-Hot Encoded Sex and Embarked
- Detected and removed multicollinear feature (family_size)
- Validated final dataset with Pandera schema

## Dataset
[Titanic Dataset — Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

## Tech stack
Python, Pandas, NumPy, Pandera, Matplotlib, Seaborn