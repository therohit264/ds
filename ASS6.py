import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
df = sns.load_dataset("iris")

print(df)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert species into numeric
df["species"] = df["species"].astype("category").cat.codes
df

# Remove outliers using IQR
Q1 = df["sepal_width"].quantile(0.25)
Q3 = df["sepal_width"].quantile(0.75)

IQR = Q3 - Q1

df = df[
    (df["sepal_width"] >= Q1 - 1.5*IQR) &
    (df["sepal_width"] <= Q3 + 1.5*IQR)
]

from numpy import positive
# Correlation Heatmap

# values - display correlation

sns.heatmap(df.corr(), annot=True)
plt.show()

# Iris dataset has only 5 columns

# Input and Output
X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))