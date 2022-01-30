import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LogNorm
from sklearn import model_selection
import warnings
warnings.filterwarnings(action='ignore')
x_1 = np.random.multivariate_normal(mean=np.array([1,1]),cov=np.array([[1,0],[0,1]]),size = 50)
x_2 = np.random.multivariate_normal(mean=np.array([3,3]),cov = np.array([[1,0],[0,1]]),size = 50)
y_1 = np.zeros(50)
y_2 = np.ones(50)
old_center = np.array([[-1,-1],[3,3]])
x_all = np.r_[x_1,x_2]
dis_frist_point_1 = []
dis_frist_point_2 = []
for i in range(len(x_all[:,0])):
    dis_frist_point_1.append(np.linalg.norm(old_center[0] - x_all[i]))
    dis_frist_point_2.append(np.linalg.norm(old_center[1] - x_all[i]))
    dis_all =[]
    dis_all = np.c_[dis_frist_point_1,dis_frist_point_2]
    np.where(dis_all == np.max(dis_all,axis=1))
