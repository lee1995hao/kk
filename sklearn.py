import random
from numpy.linalg import inv
import numpy as np
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')
n = 100
x = np.random.random(n)
x.shape
x = x.reshape(100,1)
real_beat = np.array([2])
sig_error = np.random.random(n).reshape(100,)
y = (x*real_beat).reshape(100,) + sig_error
out_idx = np.random.randint(100,size=10)
y[out_idx] = y[out_idx] + 10
y = y.reshape(100,1)
plt.scatter(x,y)
plt.show()
x= x.reshape(-1,1)
model_liner = LinearRegression()
model_liner.fit(x,y)
first_beta1 = model_liner.coef_
first_int = model_liner.intercept_


def hard_throding(x,lam):
    if abs(x) >= lam:
        return(x - np.sign(x)*lam)
    else:
        return(0)

##将函数可以将向量带进去np.vectorize(func)(data,para)
thersholding_value = []
for i in range(4):
    thersholding_value.append(np.vectorize(hard_throding)(x= np.array([1,2,3,4,5]),lam = i))

##add acol which all = 1
new_gamma = []

x_data = np.c_[np.ones(len(x[:,0])),x]
xtx = inv(np.dot(x_data.T,x_data))##求x^top 的逆矩阵
old_gamma = y -  np.array(x*first_beta1 +  first_int)
def shen_algorithm(y,x_data,lam):
    b = 0
    global new_gamma  ###从新赋值会将new_gamma 变为局部变量所以用global函数设置为全局变量
    global old_gamma
    global new_beta
    new_gamma = []
    countiune_loop = True
    while countiune_loop == True:
        b +=1
        y_adj = y - old_gamma
        new_beta = inv(x_data.T@x_data)@x_data.T@y_adj
        new_smallr = y - x_data@new_beta
        new_gamma = np.vectorize(hard_throding)(new_smallr,lam = 2)##chucuo
        new_gamma = np.array(new_gamma).reshape(100, 1)
        loop_judge = np.linalg.norm(new_gamma-old_gamma,ord=1)
        old_gamma = new_gamma
        if loop_judge <= 0.00001:
            countiune_loop = False
    return (new_beta ,new_gamma)


shen_algorithm(y = y,x_data= x_data,lam=2)

print("hallo")

