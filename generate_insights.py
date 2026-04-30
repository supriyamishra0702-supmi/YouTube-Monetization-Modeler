import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the cleaned data you created yesterday
df = pd.read_csv('cleaned_data.csv')

# 2. Correlation Heatmap
# This shows which features (Views, Watch Time, etc.) drive Revenue
plt.figure(figsize=(12, 8))
# Only use numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap: What Drives Revenue?')
plt.savefig('correlation_heatmap.png') # Saves the image to your folder
print("✅ Correlation Heatmap saved as 'correlation_heatmap.png'")

# 3. Outlier Detection (Boxplots)
# This helps identify if any videos have extreme, unusual revenue values
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['ad_revenue_usd'])
plt.title('Outlier Detection: Ad Revenue Distribution')
plt.xlabel('Revenue (USD)')
plt.savefig('revenue_outliers.png') # Saves the image to your folder
print("✅ Boxplot saved as 'revenue_outliers.png'")

# 4. Content Insights
plt.figure(figsize=(12, 6))
sns.barplot(x='category', y='ad_revenue_usd', data=df, estimator=np.mean)
plt.xticks(rotation=45)
plt.title('Average Revenue by Category')
plt.savefig('category_insights.png')
print("✅ Category Chart saved as 'category_insights.png'")
