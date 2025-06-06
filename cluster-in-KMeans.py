import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#1. Load and Visualize the Dataset
df = pd.read_csv("customer_data.csv")
scaler = StandardScaler()
X = scaler.fit_transform(df)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA Projection of Dataset")
plt.show()

#2. Fit K-Means and Assign Cluster Labels
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

#3. Use the Elbow Method to Find Optimal K
inertia = []
K_range = range(1, 11)  # Trying different values of K

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

#4. Visualize Clusters with Color-Coding
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolors='k')
plt.title("Customer Segmentation Clusters")
plt.show()

#5. Evaluate Clustering Using Silhouette Score
score = silhouette_score(X, clusters)
print("Silhouette Score:", score)

plt.show()
