import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Veri seti
data={
    "Gözlem":[1,2,3,4,5,6,7,8],
    "Yaş":[25,34,45,23,54,60,39,42],
    "Kan Basıncı":[120,135,140,125,150,155,130,145]
    }
df=pd.DataFrame(data)

#K-means kümeleme
kmeans=KMeans(n_clusters=3, random_state=42)
df["Cluster"]=kmeans.fit_predict(df) 
print(df)

plt.scatter(df["Yaş"], df["Kan Basıncı"], c=df["Cluster"], cmap="viridis")
  
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
  
plt.xlabel("Yaş")

plt.ylabel("Kan Basıncı")
  
plt.title("K-means Kümeleme")
  
plt.legend()

plt.show()
  
