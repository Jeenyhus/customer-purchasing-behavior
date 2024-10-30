import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load process data that is cleaned that is cleaned for the the data cleaning process
df = pd.read_csv('Processed_OnlineRetail.csv')

#Get a summary of the dataset
print(df.info())
print(df.describe())

#check for missing values
print(df.isnull().sum())

#Visualize the distribution of the target variable
plt.figure(figsize=(12, 6))
sns.histplot(df['UnitPrice'], kde=True)
plt.title('Distribution of Unit Price')
plt.savefig('unit_price_distribution.png')
plt.close()

#plot the correlation of key features.
plt.figure(figsize=(12, 6))
# Select only numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

#create scatter plots, box plots, and correlation matrices to identify relationships and patterns
#between the features and the target variable
plt.figure(figsize=(12, 6))
sns.pairplot(df, kind='scatter')
plt.title('Pairplot of Features')
plt.savefig('pairplot.png')
plt.close()