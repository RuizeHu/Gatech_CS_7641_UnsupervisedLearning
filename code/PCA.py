
import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np
import pandas as pd
import time
import gc
import random
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from kneed import KneeLocator
import seaborn as sb
from sklearn.metrics import silhouette_score
from sklearn import manifold
from sklearn.mixture import GaussianMixture


class Data():

    def dataAllocation(self, path):
        # Separate out the x_data and y_data and return each
        # args: string path for .csv file
        # return: pandas dataframe, pandas series
        # -------------------------------
        # ADD CODE HERE
        df = pd.read_csv(path)
        xcols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
        ycol = ['y']
        x_data = df[xcols]
        y_data = df[ycol]
#        print(y_data[y_data.y == 1].shape[0])
 #       print(df.shape[0])
        # -------------------------------
        return x_data, y_data.values.ravel()

    def processed_data_Allocation(self, path):
    # Read the processed dataset
    # -------------------------------
        df = pd.read_csv(path)
        xcols = ["age","education","default","housing","loan","contact","month","day_of_week","campaign","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","job_blue-collar","job_entrepreneur","job_housemaid","job_management","job_retired","job_self-employed","job_services","job_student","job_technician","job_unemployed","marital_married","marital_single"]
        ycol = ['y']
        x_data = df[xcols]
        y_data = df[ycol]

        return x_data, y_data.values.ravel()

    def trainSets(self, x_data, y_data):
        # Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
        # Use the train_test_split method in sklearn with the parameter 'shuffle' set to true and the 'random_state' set to 614.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe, pandas series, pandas series
        # -------------------------------
        # ADD CODE HERE
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, shuffle=True, random_state=614)
        # -------------------------------
        return x_train, x_test, y_train, y_test

    def dataPreProcess(self, x_train, x_test):
        # Pre-process the data to standardize it, otherwise the grid search will take much longer.
        # args: pandas dataframe, pandas dataframe
        # return: pandas dataframe, pandas dataframe
        # -------------------------------
        # ADD CODE HERE
        scaler = StandardScaler()
        scaler.fit(x_train)
        scaled_x_train = scaler.transform(x_train)
        scaled_x_test = scaler.transform(x_test)
        # -------------------------------
        return scaled_x_train, scaled_x_test


fig, axs = plt.subplots(1, 4)
##################### Diabetes data #############################
dataset = Data()
data = 'data/pima-indians-diabetes.csv'
x_data, y_data = dataset.dataAllocation(data)
x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
x_train_scaled, x_test_scaled = dataset.dataPreProcess(x_train, x_test)

# PCA with DS1
start = time.time()
pca = PCA()
pca.fit(x_train_scaled)
end = time.time()
print("PCA for DS1 time:", end-start)
axs[0].bar(range(1,9), pca.explained_variance_ratio_)
axs[0].set_title('DS1 component importance')
axs[0].set_xlabel('Component number')
axs[0].set_ylabel('Percentage of explained variance')

error = []
for i in range(1,9):
    pca = PCA(n_components=i)
    pca.fit(x_train_scaled)
    x_train_scaled_pca = pca.transform(x_train_scaled)
    x_reconstructed = pca.inverse_transform(x_train_scaled_pca)
    reconstruct_error = 0
    for k in range(len(x_train_scaled)):
        for j in range(8):
            reconstruct_error += (x_train_scaled[k, j] - x_reconstructed[k, j])**2
    error.append(reconstruct_error)
axs[1].plot(range(1,9), error)
axs[1].set_title('DS1 SSE after reconstruction')
axs[1].set_xlabel('Number of components')
axs[1].set_ylabel('SSE')

##################### Bank marketing data #############################
dataset = Data()
data = 'data/bank_marketing.csv'
x_data, y_data = dataset.processed_data_Allocation(data)
x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
x_train_scaled, x_test_scaled = dataset.dataPreProcess(x_train, x_test)

# PCA with DS2
start = time.time()
pca = PCA()
pca.fit(x_train_scaled)
end = time.time()
print("PCA for DS2 time:", end-start)
axs[2].bar(range(1,29), pca.explained_variance_ratio_)
axs[2].set_title('DS2 component importance')
axs[2].set_xlabel('Component number')
axs[2].set_ylabel('Percentage of explained variance')

error = []
for i in range(1,29):
    pca = PCA(n_components=i)
    pca.fit(x_train_scaled)
    x_train_scaled_pca = pca.transform(x_train_scaled)
    x_reconstructed = pca.inverse_transform(x_train_scaled_pca)
    reconstruct_error = 0
    for k in range(len(x_train_scaled)):
        for j in range(28):
            reconstruct_error += (x_train_scaled[k, j] - x_reconstructed[k, j])**2
    error.append(reconstruct_error)
axs[3].plot(range(1,29), error)
axs[3].set_title('DS2 SSE after reconstruction')
axs[3].set_xlabel('Number of components')
axs[3].set_ylabel('SSE')

plt.show()
