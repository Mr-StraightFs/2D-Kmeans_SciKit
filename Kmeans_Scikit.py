# Include all the Relevant Libraries

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

# for the purpose of this analysis I will use the Scikit's inbuilt Database dubbed : Breast Cancer Data

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print(cancer) # Familiarizing with Data .

# General Descriptive Analysis : Understanding Data



# In case the data is too disperse (whch is the case for this dataset) , consider Scaling it
X = scale(cancer.data)
print(X)
y = cancer.target
print(y)

# Step3 : Sample the test and training subsets
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2) # 80% of the data for training  , 20% for testing

# Step4 : Determine the Number of Clusters Using the " Elbow Method"
from scipy.spatial.distance import cdist, pdist

# Pick your k range , e.g K=10
k_range = range(1,10)

# Fit the kmeans model for each n_clusters = k
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

# Step 5 : visually pick your k (#of clusters) and proceed to the final Kmeans clustering and run the Kmean model
# for this Database (Breast Cancer) , the 'best' k seems to be  k=3 "as per the elbow curve".
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(X_train)
#Step 6: Predictions and Accuracy
# labels = model.labels_
# print("Labels : ", labels)
# print("predictons :", predictions)
# print(" Accuracy : ", accuracy_score(y_test, predictions))

#Step 7 : Visualization with matplotlib
# Set min and max values and give it some padding

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = .5

# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
Z = (np.c_[xx.ravel(), yy.ravel()])
plt.figure(1)
plt.clf()
plt.imshow(Z,interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
plt.plot(X_train[:, 0], X_train[:, 1], 'k.')
plt.show()
# centroids = k_means.cluster_centers_
# inert = k_means.inertia_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=169, linewidths=3,
#            color='w', zorder=8)
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()





















