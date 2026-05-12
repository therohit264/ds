# Assignment 8 - Data Visualization 1

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Histogram - Passenger Class
plt.figure(figsize=(5,4))
sns.histplot(df["Pclass"], bins=3)
plt.title("Passenger Class Distribution")
plt.show()

# Histogram - Age
plt.figure(figsize=(5,4))
sns.histplot(df["Age"].dropna(), kde=True)
plt.title("Age Distribution")
plt.show()

# Histogram - Fare
plt.figure(figsize=(5,4))
sns.histplot(df["Fare"], bins=30)
plt.title("Fare Distribution")
plt.show()

# Joint Plot - Age vs Fare
sns.jointplot(x="Age", y="Fare", data=df)
plt.show()

# Bar Plot - Fare by Gender
plt.figure(figsize=(5,4))
sns.barplot(x="Sex", y="Fare", data=df)
plt.title("Average Fare by Gender")
plt.show()

# Count Plot - Passenger Class
plt.figure(figsize=(5,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Count by Class")
plt.show()

# Pair Plot
sns.pairplot(df[["Age", "Fare", "Pclass", "Survived"]].dropna())
plt.show()