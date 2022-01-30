##k mean with boosting and bagging
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LogNorm
from sklearn import model_selection
import warnings
warnings.filterwarnings(action='ignore')
x_1 = np.random.multivariate_normal(mean=np.array([1,1]),cov=np.array([[1,0],[0,1]]),size=50)
x_2 = np.random.multivariate_normal(mean=np.array([2,2]),cov=np.array([[1,0],[0,1]]),size=50)
y_1 = np.zeros(50)
y_2 = np.ones(50)
all_kmean_y = np.r_[y_1,y_2]
all_kmean_x = np.r_[x_1,x_2]

old_center = np.array([[-1,-1],[3,3]])
old_center.shape

def my_kmean(old_center,all_kmean_x):
    coun = True
    while coun == True:
        dis_x_1 = []
        dis_x_2 = []
        for i in range(len(all_kmean_x[:,1])):
            dis_x_1.append(np.linalg.norm(old_center[0,:] - all_kmean_x[i,:]))
            dis_x_2.append(np.linalg.norm(old_center[1,:] - all_kmean_x[i,:]))
        diff = np.array(dis_x_1) - np.array(dis_x_2)
        judge_data = []
        judge_data = [0 if x >= 0 else 1 for x in diff]
        group_1_idx = []
        group_2_idx = []
        group_1_idx = np.where(np.array(judge_data) == 0)
        group_2_idx = np.where(np.array(judge_data) == 1)
        center_1 = np.average(all_kmean_x[group_1_idx,:],axis=1)
        center_2 = np.average(all_kmean_x[group_2_idx,:],axis=1)
        new_center = np.r_[center_2,center_1]
        dif = np.linalg.norm(old_center - new_center,ord = 2)
        old_center = new_center
        if dif <= 0.0000000001:
            coun = False
        np.array(judge_data) == 0

    return old_center



###boosting bagging
x_train, x_test,y_train,y_test = model_selection.train_test_split(all_kmean_x,all_kmean_y,test_size = 0.4)
loop_number = 40
all_result = []
for j in range(loop_number):
    random_x = np.random.randint(0,len(x_train[:,0]),size=20)
    sample_all_x = x_train[random_x,]
    sample_center = my_kmean(old_center = old_center,all_kmean_x = sample_all_x)
    yanzheng_1 = []
    yanzheng_2 = []
    for i in range(len(x_test[:,0])):
        yanzheng_1.append(np.linalg.norm(x_test[i] - sample_center[0]))
        yanzheng_2.append(np.linalg.norm(x_test[i] - sample_center[1]))
    diff_sample = np.array(yanzheng_1) - np.array(yanzheng_2)
    all_result.append([1 if x > 0 else 0 for x in diff_sample])
np.array(all_result).shape
final = np.mean(np.array(all_result), axis=0)
final_result = np.array([1 if x >=0.5 else 0 for x in final])
test_error_bagging = np.sum(abs(y_test - final_result))/len(y_test)





range(len(all_kmean_x[:,0]))###产生连续数
np.arange(0,100,1)
prob_p = np.ones(len(x_train[:,0]))/len(x_train[:,0])


sample_boosting = np.random.choice(range(len(x_train[:,0])),size=20,p=prob_p)
sample_y = y_train[sample_boosting]
sample_x = x_train[sample_boosting]
new_center = my_kmean(old_center= old_center,all_kmean_x = sample_x)
dis_1 = []
dis_2 = []
for i in range(len(x_train[:,0])):
    dis_1.append(np.linalg.norm(new_center[0] - x_train[i]))
    dis_2.append(np.linalg.norm(new_center[1] - x_train[i]))
diff_bossting = np.array(dis_1) - np.array(dis_2)
bossting_result = [1 if x > 0 else 0 for x in diff_bossting]
bossting_result - y_train
weight_change = [1 if x != 0 else 0 for x in bossting_result - y_train]
weight_should_shange = np.nonzero(weight_change)##查找不是为0的sample
prob_p[weight_should_shange] = np.array([0.2])
test_center = my_kmean(old_center = new_center,all_kmean_x = x_test)
