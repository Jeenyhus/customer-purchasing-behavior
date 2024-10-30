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


# Step 6: Visualize the 'Country' distribution
# Count plots are useful for categorical data to show the frequency of each category.
# This plot helps understand the distribution of customers across different countries.
plt.figure(figsize=(12, 6))
sns.countplot(y='Country', data=df, order=df['Country'].value_counts().index)
plt.title('Distribution of Customers by Country')
plt.ylabel('Country')
plt.xlabel('Number of Customers')
plt.savefig('country_distribution.png')
plt.close()

# Step 7: Visualize the 'Quantity' distribution by 'Country'
# Box plots are useful for comparing distributions of a numerical variable across different categories.
# Here, we compare the quantity of items ordered across different countries.
plt.figure(figsize=(12, 6))
sns.boxplot(x='Quantity', y='Country', data=df)
plt.title('Distribution of Quantity by Country')
plt.xlabel('Quantity')
plt.ylabel('Country')
plt.savefig('quantity_by_country.png')
plt.close()

# Step 8: Visualize the 'UnitPrice' distribution by 'Country'
# Similar to the previous step, we compare the unit prices across different countries using box plots.
plt.figure(figsize=(12, 6))
sns.boxplot(x='UnitPrice', y='Country', data=df)
plt.title('Distribution of Unit Price by Country')
plt.xlabel('Unit Price')
plt.ylabel('Country')
plt.savefig('unit_price_by_country.png')
plt.close()

# Step 9: Visualize the 'Quantity' distribution by 'InvoiceDate'
# Line plots are useful for visualizing trends over time.
# Here, we plot the quantity of items ordered over time to identify any patterns or seasonality.
plt.figure(figsize=(12, 6))
# Convert 'InvoiceDate' to datetime format for time-based plotting
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# Extract the date part for grouping by date
df['Date'] = df['InvoiceDate'].dt.date
# Group by date and sum the quantities
daily_quantity = df.groupby('Date')['Quantity'].sum()
daily_quantity.plot()
plt.title('Daily Quantity Trend')
plt.xlabel('Date')
plt.ylabel('Total Quantity')

# Save the plot for future reference
plt.savefig('daily_quantity_trend.png')
# Close the plot to free up memory
plt.close()


# Step 10: Visualize the 'UnitPrice' distribution by 'InvoiceDate'
# Similar to the previous step, we plot the unit prices over time to identify any trends or patterns.
plt.figure(figsize=(12, 6))
# Group by date and calculate the average unit price
daily_unit_price = df.groupby('Date')['UnitPrice'].mean()
daily_unit_price.plot()
plt.title('Daily Unit Price Trend')
plt.xlabel('Date')
plt.ylabel('Average Unit Price')
plt.savefig('daily_unit_price_trend.png')
plt.close()

# Step 11: Save the cleaned dataset to disk for future analysis
# After exploring and visualizing the data, it's essential to save the cleaned dataset for future use.
# We save the dataset in CSV format to preserve the changes made during cleaning and exploration.
df.to_csv('Cleaned_OnlineRetail.csv', index=False)
print("Cleaned dataset saved to 'Cleaned_OnlineRetail.csv'.")
# The output of this script will be a series of visualizations saved as image files and a cleaned dataset saved as a CSV file.
# These outputs can be used for further analysis, modeling, or reporting purposes.
# The visualizations provide insights into the data distribution, relationships between features, and trends over time.
# The cleaned dataset can be used directly for machine learning modeling or further analysis without repeating the cleaning steps.



