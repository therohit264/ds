import pandas as pd

df= pd.read_csv("Placement_data_full_class.csv")
print('Complete Dataset!')
print(df)

print("n random samples")
print(df.sample(4))

print("n last rows")
print(df.tail(5))

print("first n rows")
print(df.head(5))

print("Find missing values")
missing_val = df.isna().sum()
print(missing_val)

print("Initial statisuics")
initial_staic = df.describe()
print(initial_staic)

print("variables description")
vr_desc = df.dtypes
print(vr_desc)

print("Check Data Frame Dimensions")
dimensions = df.shape
print(dimensions)

print(df.isna())

print("proper datatype conversion")
df['sl_no']= df['sl_no'].astype(float)
print(df['sl_no'])

print('Convert categorial varaible into numeric')
df= pd.get_dummies(df, columns=['gender'])
print(df)

print(df.isna())
print(df.isna().sum())

print(df.isnull())
print(df.isnull().sum())

df = df.replace(['M','F'], ['xy','xx'])
print(df)