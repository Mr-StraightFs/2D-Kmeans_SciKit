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

# Descriptive Analysis


# In case the data is too disperse , consider Scaling it
X = scale(cancer.data)
print(X)

y = cancer.target
print(y)

# Step3 : Sampling the test and training subsets
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)














