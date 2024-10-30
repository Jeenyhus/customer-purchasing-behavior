# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load processed data that has undergone previous cleaning steps
# The dataset is assumed to be in CSV format.
# We use pd.read_csv() here as it’s optimized for loading CSV data directly into a DataFrame.
df = pd.read_csv('Processed_OnlineRetail.csv')

# Step 1: Get an overview of the dataset structure
# df.info() provides a concise summary of the DataFrame, showing column names, non-null counts, and data types.
# This helps identify any data issues such as missing values or unexpected data types early on.
print("Dataset Information:")
# Outputs details about data types and missing values
print(df.info())  
print("\nSummary Statistics:")
# Gives basic statistical summary of numerical columns (mean, std, min, etc.)
print(df.describe())  

# Step 2: Check for missing values in each column
# Missing values can interfere with analysis, so it's essential to know if they exist and handle them accordingly.
# This line gives a count of missing values per column to determine if any data cleaning is still needed.
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Step 3: Visualize the distribution of the 'UnitPrice' column
# Histograms are a great way to visualize the distribution of a single variable.
# Here, we use it to understand the spread of unit prices and detect any skewness or outliers.
plt.figure(figsize=(12, 6))
# kde=True adds a smooth curve (Kernel Density Estimation) to show distribution shape
sns.histplot(df['UnitPrice'], kde=True)  
plt.title('Distribution of Unit Price')
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
 # Save the plot for future reference
plt.savefig('unit_price_distribution.png')
# Close the plot to free up memory 
plt.close()  

# Step 4: Plot the correlation heatmap of numerical features
# Correlation helps understand relationships between numerical variables.
# A correlation matrix shows which variables move together, which is crucial in feature selection and avoiding multicollinearity.
# We only include numeric columns for correlation calculation.
plt.figure(figsize=(12, 6))
 # Select only numeric columns to avoid errors in correlation calculation
numeric_df = df.select_dtypes(include=['number']) 
# annot=True adds correlation values to cells
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')  
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# Step 5: Create pair plots to identify relationships between features
# Pair plots show scatter plots between each pair of features and can highlight relationships.
# Useful for detecting linear and nonlinear patterns and for observing distributions across features.
plt.figure(figsize=(12, 6))
# scatter plot between features; can also use kind='reg' for regression line
sns.pairplot(df, kind='scatter')  
# Adjust y position to avoid overlap with titles
plt.suptitle('Pairplot of Features', y=1.02)  
plt.savefig('pairplot.png')
plt.close()
