
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
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import KernelPCA


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


##################### Diabetes data #############################
dataset = Data()
data = 'data/pima-indians-diabetes.csv'
x_data, y_data = dataset.dataAllocation(data)
x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
x_train_scaled, x_test_scaled = dataset.dataPreProcess(x_train, x_test)

# Choose the best number of clusters using elbow method
kmeans_kwargs = {'init': 'random', 'n_init':10, 'max_iter':100, 'random_state':42, 'algorithm':'full',}

#pca = PCA(n_components=6)
#pca.fit(x_train_scaled)
#x_train_scaled_reduced = pca.transform(x_train_scaled)

#ica = FastICA(n_components=6)
#x_train_scaled_reduced = ica.fit_transform(x_train_scaled)

#rng = np.random.RandomState(42)
#rp = GaussianRandomProjection(n_components=6, random_state=rng)
#x_train_scaled_reduced = rp.fit_transform(x_train_scaled)

kpca = KernelPCA(n_components = 6, kernel='poly')
kpca.fit(x_train_scaled)
x_train_scaled_reduced = kpca.transform(x_train_scaled)

sse = []
for k in range(1,30):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x_train_scaled_reduced)
    sse.append(kmeans.inertia_)

kl = KneeLocator(range(1, 30), sse, curve="convex", direction="decreasing")
plt.figure(5)
plt.plot(range(1,30),sse)
plt.xticks(range(1,30,5))
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('DS1')
plt.show()

silhouette_coefficients = []
for k in range(2,30):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x_train_scaled_reduced)
    score = silhouette_score(x_train_scaled_reduced, kmeans.labels_)
    silhouette_coefficients.append(score)
plt.figure(6)
plt.plot(range(2,30),silhouette_coefficients)
plt.xticks(range(2,30,5))
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('DS1')
plt.show()


##################### Bank marketing data #############################
dataset = Data()
data = 'data/bank_marketing.csv'
x_data, y_data = dataset.processed_data_Allocation(data)
x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
x_train_scaled, x_test_scaled = dataset.dataPreProcess(x_train, x_test)

# Choose the best number of clusters using elbow method
kmeans_kwargs = {'init': 'random', 'n_init':10, 'max_iter':100, 'random_state':42, 'algorithm':'full',}

#pca = PCA(n_components=18)
#pca.fit(x_train_scaled)
#x_train_scaled_reduced = pca.transform(x_train_scaled)

#ica = FastICA(n_components=18)
#x_train_scaled_reduced = ica.fit_transform(x_train_scaled)

#rng = np.random.RandomState(42)
#rp = GaussianRandomProjection(n_components=18, random_state=rng)
#x_train_scaled_reduced = rp.fit_transform(x_train_scaled)

kpca = KernelPCA(n_components = 18, kernel='poly')
kpca.fit(x_train_scaled)
x_train_scaled_reduced = kpca.transform(x_train_scaled)

sse = []
for k in range(1,30):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x_train_scaled_reduced)
    sse.append(kmeans.inertia_)

kl = KneeLocator(range(1, 30), sse, curve="convex", direction="decreasing")
plt.figure(7)
plt.plot(range(1,30),sse)
plt.xticks(range(1,30,5))
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('DS2')
plt.show()

silhouette_coefficients = []
for k in range(2,30):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(x_train_scaled_reduced)
    score = silhouette_score(x_train_scaled_reduced, kmeans.labels_)
    silhouette_coefficients.append(score)
plt.figure(8)
plt.plot(range(2,30),silhouette_coefficients)
plt.xticks(range(2,30,5))
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('DS2')
plt.show()
