
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
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve


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


_, axes = plt.subplots(1, 5)
##################### Bank marketing data #############################
dataset = Data()
#data = 'data/bank_marketing.csv'
#x_data, y_data = dataset.processed_data_Allocation(data)
data = 'data/pima-indians-diabetes.csv'
x_data, y_data = dataset.dataAllocation(data)
#x_train, x_test, y_train, y_test = dataset.trainSets(x_data, y_data)
#x_train_scaled, x_test_scaled = dataset.dataPreProcess(x_train, x_test)
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = MLPClassifier(
            hidden_layer_sizes=(3), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=10000, random_state=0)

# Original data
start = time.time()
train_sizes, train_scores, test_scores, fit_times, _ = \
    learning_curve(estimator, x_data, y_data, cv=cv, n_jobs=4,
                    train_sizes=np.linspace(.1, 1.0, 5),return_times=True)
end = time.time()
print("Original time", end-start)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

axes[0].set_title("NN - original data")
axes[0].set_xlabel("Training examples")
axes[0].set_ylabel("Score")

axes[0].set_xlabel("Numer of training samples")
axes[0].set_ylabel("Score")
axes[0].grid()
axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
axes[0].legend(loc="best")
axes[0].set_ylim(0.6,1.0)
axes[0].set_title("Original data")

# Clustering reduced
x_data, y_data = dataset.dataAllocation(data)
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
kmeans_kwargs = {'init': 'random', 'n_init':10, 'max_iter':100, 'random_state':42, 'algorithm':'full',}
kmeans = KMeans(n_clusters=5, **kmeans_kwargs)
label = kmeans.fit(x_data).labels_

x_data_cl = []
pp = x_data.tolist()
for i in range(len(x_data)):
    temp = []
    temp.append(label[i])
    x_data_cl.append(temp)
x_data_cl = np.array(x_data_cl)
df = pd.DataFrame(x_data_cl, columns = ['cluster'])
x_data = pd.get_dummies(df, prefix='cluster', columns=['cluster'], drop_first=True)  
x_data = x_data.to_numpy()

start = time.time()
train_sizes, train_scores, test_scores, fit_times, _ = \
    learning_curve(estimator, x_data, y_data, cv=cv, n_jobs=4,
                    train_sizes=np.linspace(.1, 1.0, 5),return_times=True)
end = time.time()
print("Clustering time", end-start)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

axes[1].set_title("Clustering")
axes[1].set_xlabel("Training examples")
axes[1].set_ylabel("Score")

axes[1].set_xlabel("Numer of training samples")
axes[1].set_ylabel("Score")
axes[1].grid()
axes[1].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
axes[1].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
axes[1].plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
axes[1].plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
axes[1].legend(loc="best")
axes[1].set_ylim(0.6,1.0)

# Clustering + original data
x_data, y_data = dataset.dataAllocation(data)
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
kmeans_kwargs = {'init': 'random', 'n_init':10, 'max_iter':100, 'random_state':42, 'algorithm':'full',}
kmeans = KMeans(n_clusters=5, **kmeans_kwargs)
label = kmeans.fit(x_data).labels_

x_data_cl = []
pp = x_data.tolist()
for i in range(len(x_data)):
    temp = pp[i]
    temp.append(label[i])
    x_data_cl.append(temp)
x_data_cl = np.array(x_data_cl)
df = pd.DataFrame(x_data_cl, columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'cluster'])
x_data = pd.get_dummies(df, prefix='cluster', columns=['cluster'], drop_first=True)  
x_data = x_data.to_numpy()

start = time.time()
train_sizes, train_scores, test_scores, fit_times, _ = \
    learning_curve(estimator, x_data, y_data, cv=cv, n_jobs=4,
                    train_sizes=np.linspace(.1, 1.0, 5),return_times=True)
end = time.time()
print("Clustering + original data time", end-start)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

axes[2].set_title("Clustering + original data")
axes[2].set_xlabel("Training examples")
axes[2].set_ylabel("Score")

axes[2].set_xlabel("Numer of training samples")
axes[2].set_ylabel("Score")
axes[2].grid()
axes[2].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
axes[2].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
axes[2].plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
axes[2].plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
axes[2].legend(loc="best")
axes[2].set_ylim(0.6,1.0)

# PCA
x_data, y_data = dataset.dataAllocation(data)
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
pca = PCA(n_components=6)
pca.fit(x_data)
x_data_reduced = pca.transform(x_data)
start = time.time()
train_sizes, train_scores, test_scores, fit_times, _ = \
    learning_curve(estimator, x_data_reduced, y_data, cv=cv, n_jobs=4,
                    train_sizes=np.linspace(.1, 1.0, 5),return_times=True)
end = time.time()
print("PCA time", end-start)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

axes[3].set_title("PCA")
axes[3].set_xlabel("Training examples")
axes[3].set_ylabel("Score")

axes[3].set_xlabel("Numer of training samples")
axes[3].set_ylabel("Score")
axes[3].grid()
axes[3].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
axes[3].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
axes[3].plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
axes[3].plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
axes[3].legend(loc="best")
axes[3].set_ylim(0.6,1.0)

# Clustering + PCA
x_data, y_data = dataset.dataAllocation(data)
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
pca = PCA(n_components=6)
pca.fit(x_data)
x_data_reduced = pca.transform(x_data)
kmeans_kwargs = {'init': 'random', 'n_init':10, 'max_iter':100, 'random_state':42, 'algorithm':'full',}
kmeans = KMeans(n_clusters=5, **kmeans_kwargs)
label = kmeans.fit(x_data).labels_

x_data_cl = []
pp = x_data_reduced.tolist()
for i in range(len(x_data)):
    temp = pp[i]
    temp.append(label[i])
    x_data_cl.append(temp)
x_data_cl = np.array(x_data_cl)
df = pd.DataFrame(x_data_cl, columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'cluster'])
x_data = pd.get_dummies(df, prefix='cluster', columns=['cluster'], drop_first=True)  
x_data = x_data.to_numpy()

start = time.time()
train_sizes, train_scores, test_scores, fit_times, _ = \
    learning_curve(estimator, x_data, y_data, cv=cv, n_jobs=4,
                    train_sizes=np.linspace(.1, 1.0, 5),return_times=True)
end = time.time()
print("Clustering + PCA time", end-start)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

axes[4].set_title("Clustering + PCA")
axes[4].set_xlabel("Training examples")
axes[4].set_ylabel("Score")

axes[4].set_xlabel("Numer of training samples")
axes[4].set_ylabel("Score")
axes[4].grid()
axes[4].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
axes[4].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
axes[4].plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
axes[4].plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
axes[4].legend(loc="best")
axes[4].set_ylim(0.6,1.0)

plt.show()