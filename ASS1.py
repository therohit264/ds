import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('Placement_data_full_class.csv')  
print(df.head())
print("\n\n")

# Mapping
df['status'] = df['status'].map({'Not Placed': 0, 'Placed': 1})
print(df['status'])

print("\n\n")
# Normalization
df['salary'] = StandardScaler().fit_transform(df[['salary']])
print("Z-Score:\n", df['salary'])

print("\n\n")
# Normalization
df['salary'] = MinMaxScaler().fit_transform(df[['salary']])
print(df['salary'])

print("\n\n")
# Handeled null values
print(df.isnull().sum())
print("\n\n")

df.fillna(df.mean(numeric_only=True), inplace=True)   # fill numeric with mean
df.fillna(df.mode().iloc[0], inplace=True)             # fill categorical with mode
print(df.isnull().sum())

print("\n\n")
print(df)