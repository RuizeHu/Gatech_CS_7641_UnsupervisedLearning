
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


fig, axs = plt.subplots(2, 3)
##################### Diabetes data #############################
dataset = Data()
data = 'data/pima-indians-diabetes.csv'
x_data, y_data = dataset.dataAllocation(data)
x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
x_train_scaled, x_test_scaled = dataset.dataPreProcess(x_train, x_test)

# KM with DS1
kmeans_kwargs = {'init': 'random', 'n_init':10, 'max_iter':100, 'random_state':42, 'algorithm':'full',}
start = time.time()
kmeans = KMeans(n_clusters=4, **kmeans_kwargs)
label = kmeans.fit(x_train_scaled).labels_
end = time.time()
print("KM with DS1 time:", end-start)
tsne_transform = manifold.TSNE(n_components=2, perplexity=100, init='pca', random_state=42)
feature2D_DS1 = tsne_transform.fit_transform(x_train_scaled)
axs[0, 0].scatter(feature2D_DS1[:,0], feature2D_DS1[:,1], c=label, cmap=plt.cm.Spectral, s=5)
axs[0, 0].set_title('Clusters for DS1 with KM')

# EM with DS1
em_kwargs = {'covariance_type': 'full', 'n_init':10, 'max_iter':100, 'random_state':42, 'init_params':'kmeans'}
start = time.time()
em = GaussianMixture(n_components=5, **em_kwargs)
label = em.fit(x_train_scaled).predict(x_train_scaled)
end = time.time()
print("EM with DS1 time:", end-start)
tsne_transform = manifold.TSNE(n_components=2, perplexity=100, init='pca', random_state=42)
feature2D_DS1 = tsne_transform.fit_transform(x_train_scaled)
axs[0, 1].scatter(feature2D_DS1[:,0], feature2D_DS1[:,1], c=label, cmap=plt.cm.Spectral, s=5)
axs[0, 1].set_title('Clusters for DS1 with EM')

axs[0, 2].scatter(feature2D_DS1[:,0], feature2D_DS1[:,1], c=y_train, cmap=plt.cm.Spectral, s=5)
axs[0, 2].set_title('Clusters for DS1 - True Label')


dataset = Data()
data = 'data/bank_marketing.csv'
x_data, y_data = dataset.processed_data_Allocation(data)
x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
x_train_scaled, x_test_scaled = dataset.dataPreProcess(x_train, x_test)

# KM with DS2
kmeans_kwargs = {'init': 'random', 'n_init':10, 'max_iter':100, 'random_state':42, 'algorithm':'full',}
start = time.time()
kmeans = KMeans(n_clusters=6, **kmeans_kwargs)
label = kmeans.fit(x_train_scaled).labels_
end = time.time()
print("KM with DS2 time:", end-start)
tsne_transform = manifold.TSNE(n_components=2, perplexity=100, init='pca', random_state=42)
feature2D_DS2 = tsne_transform.fit_transform(x_train_scaled)
axs[1, 0].scatter(feature2D_DS2[:,0], feature2D_DS2[:,1], c=label, cmap=plt.cm.Spectral, s=5)
axs[1, 0].set_title('Clusters for DS2 with KM')

# EM with DS2
em_kwargs = {'covariance_type': 'full', 'n_init':10, 'max_iter':100, 'random_state':42, 'init_params':'kmeans'}
start = time.time()
em = GaussianMixture(n_components=5, **em_kwargs)
label = em.fit(x_train_scaled).predict(x_train_scaled)
end = time.time()
print("EM with DS2 time:", end-start)
tsne_transform = manifold.TSNE(n_components=2, perplexity=100, init='pca', random_state=42)
feature2D_DS2 = tsne_transform.fit_transform(x_train_scaled)
axs[1, 1].scatter(feature2D_DS2[:,0], feature2D_DS2[:,1], c=label, cmap=plt.cm.Spectral, s=5)
axs[1, 1].set_title('Clusters for DS2 with EM')

axs[1, 2].scatter(feature2D_DS2[:,0], feature2D_DS2[:,1], c=y_train, cmap=plt.cm.Spectral, s=5)
axs[1, 2].set_title('Clusters for DS2 - True Label')


fig.tight_layout()


plt.show()
