import pandas as pd
df = pd.read_csv("titanic_data.csv") 
print(df.head())

#missing values 
print(df.isnull().sum())

#filling the missing value 
df['Age'] = df['Age'].fillna(df['Age'].mean())

#see there any missing values
print(df.isnull().sum())

#draw the histogram of 1-variable, 2-variable,3-variable
#1-variable Age and fare
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data=df, x='Age')
plt.show()
sns.histplot(data=df, x='Fare')
plt.show()
sns.histplot(data=df, x='Age', hue='Sex', multiple="dodge", shrink=8)
plt.show()
sns.histplot(data=df, x='Age', hue="Pclass", multiple ="dodge", shrink=8)
plt.show()