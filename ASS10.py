# Assignment 10 - Data Visualization 3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load inbuilt iris dataset
df = sns.load_dataset("iris")

print(df.head())

print("\nData Types:")
print(df.dtypes)

# Histogram
sns.histplot(df["sepal_length"], kde=True)
plt.title("Sepal Length Distribution")
plt.show()

# Box Plot
sns.boxplot(x="species", y="petal_length", data=df)
plt.title("Petal Length by Species")
plt.show()

# Pair Plot
sns.pairplot(df, hue="species")
plt.show()

# Observations
print("\nObservations:")
print("- Setosa has smaller petal length")
print("- Virginica has larger petal length")
print("- Features are useful for classification")