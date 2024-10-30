import pandas as pd
from scipy import stats
import numpy as np

# Load the data in chunks to reduce memory usage
# Why? To avoid loading the entire dataset into memory at once
chunk_size = 10000 

# Load the data from the CSV file
file_path = 'OnlineRetail.csv'
try:
    # Load the data in chunks to reduce memory usage and avoid memory errors
    df_chunks = pd.read_csv(file_path, encoding='ISO-8859-1', chunksize=chunk_size)
    df = pd.concat(df_chunks)
    # Reset the index after concatenating the chunks
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{file_path}' file not found.")
    exit()

# Identify initial missing values
initial_missing = df.isnull().sum().sum()
print(f"Initial total missing values: {initial_missing}")

# Optimize data types for memory efficiency
#` Why? To reduce memory usage and improve performance
for col in df.select_dtypes(include='float').columns:
    df[col] = df[col].astype('float32')
for col in df.select_dtypes(include='int').columns:
    df[col] = df[col].astype('int32')
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

# Impute missing values
# Fill numeric columns with their mean if missing values exist
# Why? To replace missing values with a central value that preserves the distribution
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()) if x.isnull().any() else x)

# Fill non-numeric columns with their mode if missing values exist
# Why? To replace missing values with the most frequent value
for col in df.select_dtypes(exclude='number').columns:
    if df[col].isnull().any():
        mode_value = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_value)

# Forward and backward fill, if necessary
# Why? To fill missing values with the nearest non-missing value
if df.isnull().sum().sum() > 0:
    df = df.ffill().bfill()

# Interpolate to fill remaining missing values in numeric columns
# Why? To estimate missing values based on the values of neighboring data points
if df.select_dtypes(include='number').isnull().sum().sum() > 0:
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

# Drop any remaining rows and columns with missing values
# Why? To ensure the data is clean and ready for analysis
df.dropna(axis=0, inplace=True)
df.dropna(axis=1, inplace=True)

final_missing = df.isnull().sum().sum()
print(f"Remaining missing values after imputation and dropping: {final_missing}")

# Detect and remove outliers using Z-score
# Why? To remove extreme values that can skew the model
z_scores = np.abs(stats.zscore(df.select_dtypes(include='number')))
outliers = (z_scores > 3).any(axis=1)
initial_shape = df.shape
df = df[~outliers]
print(f"Outliers removed: {initial_shape[0] - df.shape[0]} rows")

# Winsorization: Replace outliers with the 1st and 99th percentile values
# Why? To reduce the impact of outliers on the model without removing them entirely
for col in numeric_cols:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
print("Winsorization applied to numerical columns.")

# Apply logarithmic transformation to reduce the effect of outliers, after ensuring no negatives
# Why? To make the data more normally distributed and reduce the impact of outliers
for col in numeric_cols:
    if (df[col] <= 0).any():
        min_positive_value = df[col][df[col] > 0].min()
        df[col] = df[col].apply(lambda x: x if x > 0 else min_positive_value)
    df[col] = np.log1p(df[col])

print("Logarithmic transformation applied to numeric columns.")

# Standardize date format in 'InvoiceDate' column, if it exists
# Why? To ensure consistency in date format for analysis
if 'InvoiceDate' in df.columns:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df.dropna(subset=['InvoiceDate'], inplace=True)
    print("Date format standardized in 'InvoiceDate' column.")
else:
    print("Warning: 'InvoiceDate' column not found. Skipping date standardization.")

# Selectively apply one-hot encoding on categorical columns with a limited number of unique values
# Why? To convert categorical data into numerical format for machine learning models
max_categories = 20
categorical_cols = [col for col in df.select_dtypes(include='category').columns if df[col].nunique() <= max_categories]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("One-hot encoding applied to categorical columns.")

# Check for duplicates and remove them
# Why? To ensure data integrity and avoid bias in the analysis
initial_shape = df.shape
df.drop_duplicates(inplace=True)
# Calculate the number of duplicates removed
duplicates_removed = initial_shape[0] - df.shape[0]
print(f"Duplicates removed: {duplicates_removed}")

# Save processed data to disk to free memory in case of further analysis
# Why? To save the cleaned data for future use without having to repeat the preprocessing steps
processed_file_path = 'Processed_OnlineRetail.csv'
df.to_csv(processed_file_path, index=False)
print(f"Data preprocessing complete. Processed data saved to '{processed_file_path}'.")
