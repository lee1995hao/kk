##k mean with boosting and bagging
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LogNorm
from sklearn import model_selection
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings(action='ignore')
x_1 = np.random.multivariate_normal(mean=np.array([1,1]),cov=np.array([[1,0],[0,1]]),size=50)
x_2 = np.random.multivariate_normal(mean=np.array([2,2]),cov=np.array([[1,0],[0,1]]),size=50)
y_1 = np.zeros(50)
y_2 = np.ones(50)
all_kmean_y = np.r_[y_1,y_2]
all_kmean_x = np.r_[x_1,x_2]
model_normal_kmean = KMeans(n_clusters=2, random_state=0).fit(all_kmean_x)


def okm(lam,model_normal_kmean,all_kmean_x):
    group1_idx = np.where(model_normal_kmean.labels_ == 1)
    group0_idx = np.where(model_normal_kmean.labels_ == 0)
    group0_number = all_kmean_x[group0_idx]
    group1_number = all_kmean_x[group1_idx]
    group0_center = model_normal_kmean.cluster_centers_[0]
    group1_center = model_normal_kmean.cluster_centers_[1]
    group1_r = []
    for i in range(len(group1_number[:,0])):
        group1_r.append(np.linalg.norm(group1_number[i] - group1_center))
    group0_r = []
    for i in range(len(group0_number[:, 0])):
        group0_r.append(np.linalg.norm(group0_number[i] - group0_center))
    lam = 1
    group0_gamma = []
    for i in range(len(group0_r)):
        group0_gamma.append(max(0,1 - lam/group0_r[i]))
    group1_gamma = []
    for i in range(len(group1_r)):
        group1_gamma.append(max(0,1 - lam/group1_r[i]))
    group1_e = (group1_number - group1_center) * np.array(group1_gamma).reshape(len(group1_gamma),1)
    group0_e = (group0_number - group0_center) * np.array(group0_gamma).reshape(len(group0_gamma),1)
    error_1 = np.linalg.norm((group1_number - group1_e - group1_center) + (group1_number - group1_e - group1_center)) + np.linalg.norm(group1_e + group1_e)
    error_0 = np.linalg.norm((group0_number - group0_e - group0_center) + (group0_number - group0_e - group0_center)) + np.linalg.norm(group0_e + group0_e)
    new_loss = np.sum(error_1 + error_0)
    old_loss = 0
    diff = np.abs(old_loss - new_loss)
    old_loss = new_loss
    if diff <= 0.0000000001:
        change = False