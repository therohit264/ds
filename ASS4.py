import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("HousingData.csv")

# Missing values
print(df.isnull().sum())

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Remove outliers using IQR
Q1 = df["RM"].quantile(0.25)
Q3 = df["RM"].quantile(0.75)

IQR = Q3 - Q1

df = df[
    (df["RM"] >= Q1 - 1.5*IQR) &
    (df["RM"] <= Q3 + 1.5*IQR)
]

# Heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()

# Input and Output
X = df[['RM', 'LSTAT', 'PTRATIO']]
y = df['MEDV']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("MAE:", mean_absolute_error(y_test, y_pred))
print("Model Score:", model.score(X_test, y_test))

# Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()