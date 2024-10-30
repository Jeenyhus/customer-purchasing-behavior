import pandas as pd

# Load the data
df = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')

# Display the first 5 rows of the data
# print(df.head())

# Display the shape of the data
# print(df.shape)

# Display the columns of the data
# print(df.columns)

#identify the missing values using pandas isnull() method
missing_values=pd.isnull(df)
# print(missing_values)

#impute missing values, Replace missing values with mean,mode or median of the respective columns
# Fill numeric columns with their mean
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Fill non-numeric columns with their mode
df.fillna({col: df[col].mode()[0] for col in df.select_dtypes(exclude='number').columns}, inplace=True)

# Forward/Backward fill: Fill missing values with the previous or next non-missing value in the column.
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Interpolation: Use linear or polynomial interpolation to estimate missing values.
df.interpolate(method='linear', inplace=True)

# Drop rows/columns with missing values using dropna() method of pandas
# Drop rows with missing values
df.dropna(axis=0, inplace=True)

# Drop columns with missing values
df.dropna(axis=1, inplace=True)
