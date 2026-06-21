import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#step1 : loading dataset 
df=pd.read_csv("Titanic-Dataset.csv")

#step2 : dataset info
print("\nShape of dataset: ")
print(df.shape)
print("\nFirst 5 rows: ")
print(df.head())
print("\ncolumn info: ")
print(df.info())
print("\nBasic statistics: ")
print(df.describe())

#step3 : finding missing values
print("\nMissing values per column:")
missing = df.isnull().sum()                    
missing_percent = (missing / len(df)) * 100   
print(pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_percent
}))

#step4 : handle missing values using decision matrix given

#too many missing values (77%)
df = df.drop(columns=['Cabin'])
print("\nDropped 'Cabin' column. New shape:", df.shape)

#very few missing (0.2%)
df = df.dropna(subset=['Embarked'])
print("\nDropped rows with missing 'Embarked'. New shape:", df.shape)

#moderate missing (20%)
age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)
print("\nFilled missing 'Age' values with median: ",age_median)


print("\nMissing values after cleaning:")
print(df.isnull().sum())

#outliers visualised
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(y=df['Age'], ax=axes[0])
axes[0].set_title('Age - Outliers Check')

sns.boxplot(y=df['Fare'], ax=axes[1])
axes[1].set_title('Fare - Outliers Check')

plt.tight_layout()
plt.savefig('outliers_check.png')
print("\nSaved boxplots as 'outliers_check.png' - check your folder!")

#ste6: find IQR

def cap_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)   
    Q3 = dataframe[column].quantile(0.75)   
    IQR = Q3 - Q1                            

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"\n{column}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    print(f"{column}: Lower bound={lower_bound:.2f}, Upper bound={upper_bound:.2f}")

    #Count outliers 
    outliers_count = ((dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)).sum()
    print(f"{column}: Number of outliers found = {outliers_count}")

    dataframe[column] = np.clip(dataframe[column], lower_bound, upper_bound)

    return dataframe


df=cap_iqr(df, 'Age')
df=cap_iqr(df, 'Fare')

print("\nOutliers capped! New stats:")
print(df[['Age', 'Fare']].describe())

#step7: feature engineering
df['family_size']=df['SibSp']+df['Parch']+1
df['is_alone']=(df['family_size']==1).astype(int)
df['fare_per_person']=df['Fare']/df['family_size']

print("\nNew features created!")
print(df[['SibSp', 'Parch', 'family_size', 'is_alone', 'Fare', 'fare_per_person']].head(10))

#step8: one hot encoding
df=pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print(df.columns.tolist())
print(df.head())

#removing columns that are not needed
df = df.drop(columns=['Name', 'Ticket', 'PassengerId'])
print(df.columns.tolist())


#correlation matrix
correlation_matrix = df.corr()
print("\n")
print(correlation_matrix)

corr_matrix = df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_pairs = [(col, row, upper_triangle.loc[row, col]) 
                    for col in upper_triangle.columns 
                    for row in upper_triangle.index 
                    if upper_triangle.loc[row, col] > 0.80]
print("\nHigh correlation pairs:", high_corr_pairs)

df = df.drop(columns=['family_size'])

import pandera.pandas as pa

schema=pa.DataFrameSchema({
    "Survived": pa.Column(int, pa.Check.isin([0, 1])),
    "Pclass": pa.Column(int, pa.Check.isin([1, 2, 3])),
    "Age": pa.Column(float, pa.Check.in_range(0, 100)),
    "SibSp": pa.Column(int, pa.Check.ge(0)),
    "Parch": pa.Column(int, pa.Check.ge(0)),
    "Fare": pa.Column(float, pa.Check.ge(0)),
    "is_alone": pa.Column(int, pa.Check.isin([0, 1])),
    "fare_per_person": pa.Column(float, pa.Check.ge(0)),
    "Sex_male": pa.Column(bool),
    "Embarked_Q": pa.Column(bool),
    "Embarked_S": pa.Column(bool),
})

try:
    schema.validate(df, lazy=True)
    print("\nData passed validation!")
except pa.errors.SchemaErrors as e:
    print("\nValidation errors found:")
    print(e.failure_cases)

df.to_csv('Titanic_Cleaned.csv', index=False)
print("\nFinal cleaned dataset saved as 'Titanic_Cleaned.csv'")
print(f"Final shape: {df.shape}")
print(df.head())