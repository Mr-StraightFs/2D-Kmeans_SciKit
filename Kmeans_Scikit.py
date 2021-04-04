# Including the Relevant Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

#Step2 : Import , Preprocess , clean and prepare Data for Analysis

# Data Importation
# In this part , Uncomment the right line of code in line the type of Data that you want to import
# I included the most common file types , namely : XSL and CSV Files
# For XSL files
# df = pd.read_excel ('Path where the Excel file is stored\File name.xlsx', sheet_name='your Excel sheet name')
# For CSV
# data = pd.read_csv('Path where the Excel file is stored\File name.xlsx' , delimiter=';')

# for the purpose of this analysis I will use the well known Scikit-inbuilt Database dubbed : Breast Cancer Data

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
# Familiarizing with Data .
print(cancer)

# General Descriptive Analysis : Understanding Data



# In case the data is too disperse , consider Scaling it
X = scale(cancer.data)
print(X)

y = cancer.target
print(y)

# Step3 : Sampling the test and training subsets
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

# Step4 : Determining the Number of Clusters Using the " Elbow Method"
from scipy.spatial.distance import cdist, pdist

# Pick your k range , e.g K=10
k_range = range(1,10)

# Fit the kmeans model for each n_clusters = k
# model = KMeans()
# model.fit(X_train)
k_means_var = [KMeans(n_clusters=k).fit(X_train) for k in k_range]

# Fetch out and save the cluster centers for each model in an array
centroids = [X.cluster_centers_ for X in k_means_var]

# Calculate the Euclidean distance from each point to its assigned cluster's center
k_eucld = [cdist(X_train, ctr, 'euclidean') for ctr in centroids]
dst = [np.min(ke,axis=1) for ke in k_eucld]

# Total within-cluster sum of squares
twcss = [sum(d**2) for d in dst]

# The total sum of squares
tss = sum(pdist(X_train)**2)/X_train.shape[0]

# The between-cluster sum of squares
bcss = tss - twcss

# Visualizing the elbow curve
grph = plt.figure().add_subplot(111)
grph.plot(k_range, bcss/tss*100, 'b*-')
grph.set_ylim((0,100))
plt.grid(True)
plt.xlabel('The Number of Clusters ')
plt.ylabel('The Percentage of variance explained')
plt.title('Variance Explained per k')
plt.show()



















