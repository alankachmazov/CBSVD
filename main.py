#This a recreation of the sb-svdpp algorithm by Nima Mirbakhsh and Charles X. Ling
#For this algorithm, the python package "surprise" is needed for the matrix factorization (SVD++) and "scikit-lean"
# for finding clusters with KMeans

from surprise import AlgoBase, Dataset, SVDpp
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


class CBSVDpp(AlgoBase):
    # Alpha should be set at the beginning
    def __init__(self, a=0.15, clusters=50, n_factors=50, n_epochs=20, init_mean=0, init_std_dev=.1,
                 lr_all=.007, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None,
                 lr_qi=None, lr_yj=None, reg_bu=None, reg_bi=None, reg_pu=None,
                 reg_qi=None, reg_yj=None, random_state=None, verbose=False):

        self.bu = None  # User bias
        self.bi = None  # Item bias
        self.pu = None  # User factors
        self.qi = None  # Item factors
        self.yj = None  # implicit item factor
        self.cat_pu = {}  # Dict with the KMean categories, user and their latent vectors
        self.cat_qi = {}  # Dict with the KMean categories, items and their latent vectors
        self.cat_yj = {}  # Dict with the KMean categories, items and their latent vectors
        self.a = a  # alpha defines how much weight should the categories get
        self.clusters = clusters  # Number of KMeans cluster
        self.factors = n_factors  # Number of factors for factorization

        ###################################################
        ## Vars for the svdpp algorithm - for Gridsearch ##
        ###################################################
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev

        self.lr_all = lr_all
        self.reg_all = reg_all

        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all

        self.random_state = random_state
        self.verbose = verbose

        AlgoBase.__init__(self)

    # the fit method is used, to train the model on the dataset.
    # The input variables: n_factors: SVD++ factors, n_clusters: KMeans how many clusters should be created
    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.cb_svdpp(trainset)
        return self

    def cb_svdpp(self, trainset):
        algo = SVDpp(n_factors=self.factors, n_epochs=self.n_epochs, init_mean=self.init_mean,
                     init_std_dev=self.init_std_dev, lr_all=self.lr_all,
                     reg_all=self.reg_all, lr_bu=self.lr_bu, lr_bi=self.lr_bi, lr_pu=self.lr_pu, lr_qi=self.lr_qi,
                     lr_yj=self.lr_yj,
                     reg_bu=self.reg_bu, reg_bi=self.reg_bi, reg_pu=self.reg_pu, reg_qi=self.reg_qi, reg_yj=self.reg_yj)

        svd_output = algo.fit(trainset)
        kmeans = KMeans(init="random", n_clusters=self.clusters, n_init=10, random_state=1)

        # Setting the class variables of the prediction
        self.bu = svd_output.bu
        self.bi = svd_output.bi
        self.qi = svd_output.qi
        self.pu = svd_output.pu
        self.yj = svd_output.yj

        # Returns two dataframes:
        # 1. with the categories for each user or item
        # 2. The mean value of each category

        df_qci, df_qci_mean = create_dataframes(self.qi, kmeans.fit(self.qi), 'items')
        df_ycj, df_ycj_mean = create_dataframes(self.yj, kmeans.fit(self.yj), 'items')
        df_pcu, df_pcu_mean = create_dataframes(self.pu, kmeans.fit(self.pu), 'users')

        # Defining the dicts with where the category is the key and the value
        # is an another dict with the mean latent vector and the internal IDs of the user/item
        self.cat_qi = create_cluster_dict(df_qci, df_qci_mean, 'items')
        self.cat_yj = create_cluster_dict(df_ycj, df_ycj_mean, 'items')
        self.cat_pu = create_cluster_dict(df_pcu, df_pcu_mean, 'users')

    def estimate(self, u, i):

        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            # Getting the cluster latent vector for the given user/item
            pcu = get_category_latent_vector(u, "users", self.cat_pu)
            qci = get_category_latent_vector(i, "items", self.cat_qi)

            # Calculating the new qi and pu with the weights of the cluster mean latent vector
            new_qi = ((1 - self.a) * self.qi[i]) + (self.a * qci[0])
            new_pu = ((1 - self.a) * self.pu[u]) + (self.a * pcu[0])

            Iu = len(self.trainset.ur[u])  # nb of items rated by u
            # Calculating the user implicit feedback
            u_impl_feedback = self.user_implicit_feedback(u)
            # calculating the dot product of qi and pu with the implicit user feedback
            est += np.dot(new_qi, (new_pu + u_impl_feedback))
        return est

    def user_implicit_feedback(self, u):
        Iu = len(self.trainset.ur[u])  # nb of items rated by u
        u_impl_feedback = (sum(((1 - self.a) * self.yj[j] +
                                self.a * get_category_latent_vector(j, "items", self.cat_yj)[0])
                               for (j, _)
                               in self.trainset.ur[u]) / np.sqrt(Iu))
        return u_impl_feedback


# Helper functions
# Create two dataframes from numpy array and from the kmeans object
def create_dataframes(numpy_array, kmeans, category):
    df = pd.DataFrame(numpy_array)
    # Creating a new row with the clusters for each user/item
    df["cluster"] = kmeans.labels_
    # Sorting is not necessary but easier to inspect
    df = df.sort_values(by=['cluster'])
    df[category] = df.index
    # Calculating the mean of each cluster
    df_mean = df.groupby('cluster').mean()
    del df_mean[category]
    return df, df_mean


# Returns a dict, with the cluster as key and as value is a dict with the users/items in each
# cluster and the mean latent vector of the cluster
def create_cluster_dict(df, df_mean, category) -> dict:
    data_dict = {}
    # Adding the clusters as key with a dict of empty user/items list and latent vector
    for index, row in df.iterrows():
        key = int(row['cluster'])
        if key not in data_dict:
            data_dict[key] = {category: [],
                              'latent_vector': None}
        data_dict[key].setdefault(category, []).append(int(row[category]))

    # Adding the user/items and the mean latent vector to the correct cluster in the dict
    for key, items in data_dict.items():
        items[category] = np.asarray(items[category])
        items['latent_vector'] = df_mean.loc[[key]].to_numpy()
    return data_dict


# Searches the category dictionary for a sepcific user/item and returns the mean latent vector of the category where
# the given user/item is in
def get_category_latent_vector(iid, category, cat_var):
    found = False
    cat_latent_vector = None
    for key, value in cat_var.items():
        for items in value[category]:
            if items == iid:
                cat_latent_vector = value['latent_vector']
                found = True
        # Not the best solution but it works
        if found:
            break
    return cat_latent_vector

# Testing the algorithm
data = Dataset.load_builtin("ml-100k")

# The algorithm can get every arguments of svd++, alpha and for KMeans the clusters
algo = CBSVDpp(a=0.20, clusters=70, n_factors=570)

trainset = data.build_full_trainset()
svd_output = algo.fit(trainset)

# Showing the first 2 user categories in the category dict
for i in range(2):
    print(f"Category: {i} \n{svd_output.cat_pu[i]}")

# User bias
user_bias = svd_output.bu
# print(user_bias)

# Item factor
item_factor = svd_output.qi
# print(item_factor)

# Predicting the rating of user 100 for the movie 77
svd_output.predict("100", "77")

# Grid search example
from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin("ml-100k")

# param_grid = {"n_factors": [100, 150], "a": [0.15, 0.10], "reg_all": [0.4, 0.6], "clusters": [50, 100]}
param_grid = {"a": [0.15, 0.10]}
gs = GridSearchCV(CBSVDpp, param_grid, measures=["rmse"], cv=2)

gs.fit(data)


algo = gs.best_estimator["rmse"]

# retrain on the whole set A
trainset = data.build_full_trainset()
algo.fit(trainset)

# Cross Validation
from surprise.model_selection import cross_validate
from surprise import Dataset
import random
import numpy as np

my_seed = 4444
random.seed(my_seed)
np.random.seed(my_seed)

data = Dataset.load_builtin("ml-100k")
algo = CBSVDpp(a=0.15, clusters=50, n_factors=50)

cv = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
cv
print(cv)


from surprise.model_selection import KFold, cross_validate
from surprise import accuracy, Dataset

#kf = KFold(n_splits=2)
data = Dataset.load_builtin("ml-100k")
algo = CBSVDpp(a=0.15, clusters=50, n_factors=50)

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
"""
for trainset, testset in kf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
"""
# Result of the last test of CBSVDpp: RMSE: 0.9395 & RMSE: 0.9355
# Result of the last test of CBSVDyjpp (yj and qi have the same clusters): RMSE: 0.9406 & RMSE: 0.9425