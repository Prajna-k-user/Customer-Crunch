# Customer Crunch - Customer Segmentation Project

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load Dataset (you can replace this with your own CSV file)
df = pd.read_csv("Mall_Customers.csv")

# Preview data
print(df.head())

# Data Preprocessing
df.rename(columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)': 'SpendingScore'
}, inplace=True)

# Selecting features for clustering
X = df[['Age', 'Income', 'SpendingScore']]

# Finding the optimal number of clusters using Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plotting the elbow graph
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig("elbow_plot.png")
plt.show()

# Apply KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Cluster Summary
print(df.groupby('Cluster')[['Age', 'Income', 'SpendingScore']].mean())

# Plotting the clusters
plt.figure(figsize=(8,5))
sns.scatterplot(x='Income', y='SpendingScore', hue='Cluster', data=df, palette='Set1')
plt.title('Customer Segments')
plt.savefig("customer_clusters.png")
plt.show()

# Save the result to a new CSV
df.to_csv("Customer_Segments.csv", index=False)

print("Project Completed and Files Saved: Customer_Segments.csv, elbow_plot.png, customer_clusters.png")
