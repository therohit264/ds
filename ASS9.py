# Assignment 9 - Data Visualization 2

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

print(df.head())

# Count Plot - Survival
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Pie Chart - Gender
df["Sex"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.title("Gender Distribution")
plt.show()

# Box Plot - Age by Gender
sns.boxplot(x="Sex", y="Age", data=df)
plt.title("Age Distribution by Gender")
plt.show()

# Cross Tabulation
ct = pd.crosstab(df["Sex"], df["Survived"])

print("\nCross Tabulation:")
print(ct)

# Heatmap
sns.heatmap(ct, annot=True, fmt='d')
plt.title("Gender vs Survival")
plt.show()