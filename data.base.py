import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings(action='ignore')



def make_data(n, mean ,cov ,weight):
  n_clu,n_fea = mean.shape
  data = np.zeros((n,n_fea))
  for i in range(n):
    k = np.random.choice(n_clu,size=1,p = weights)[0]
    x = np.random.multivariate_normal(mean[k],cov[k])
    data[i] = x
  return(data)

n = 200
mean_b = np.array([[5,2],
                 [10,3],
                 [9,8],
                 [11,10]
                 ])
cov = np.array([[[1,0],[0,1]],
                [[1,0],[0,1]],
                [[1,0],[0,1]],
                [[1,0],[0,1]]
                ])
weights_b = np.array([0.1,0.4,0.4,0.1])
mean.shape

data_blue = make_data(n = n,mean = mean ,cov = cov,weight = weights)
type(data_blue)
len(data_blue[:,1])
train_blue = data_blue[:100]



mean_o = np.array([[8,4],
                 [5,8],
                 [6,11],
                 [9,12]
                 ])
cov = np.array([[[1,0],[0,1]],
                [[1,0],[0,1]],
                [[1,0],[0,1]],
                [[1,0],[0,1]]
                ])
weights_o = np.array([0.3,0.3,0.3,0.1])
data_orange = make_data(n = n,mean = mean ,cov = cov ,weight= weights)
train_orange= data_orange[:100]
plt.rcParams["figure.figsize"] = (6,6)
plt.plot(train_orange[:,0],train_orange[:,1],color = "blue")
plt.show()


o_data = train_orange
b_data = train_blue
point = np.array([1,2])

all_data = np.concatenate((o_data,b_data),axis=0)
all_data.shape
def myknn(point,all_data,k):
  distance_all = []
  for i in range(len(all_data[:,0])):
    dis_o= np.linalg.norm(point - all_data[i,],ord = 2)
    distance_all.append(dis_o)
  rank_point = np.argsort(distance_all)[0:k]
  result = 0
  for i in range(len(rank_point)):
    if rank_point[i] > 100:
      result += 1
    else:
      result += 0
  aa = np.sum(result)
  if aa > k/2:
    return(1)
  else:
    return(0)



pro_point = []
for x1 in np.arange(0,6,0.1):
  for x2 in np.arange(0,6,0.1):
    pro_point.append([x1,x2])
pro_point = np.array(pro_point)
test_result = []
for i in range(len(pro_point[:,0])):
   test_result.append(myknn(point = pro_point[i],all_data = all_data,k= 3))



test_data_o = make_data(n= 200, mean =mean_o,cov = cov,weight= weights_o)
test_data_b = make_data(n= 200, mean =mean_b,cov = cov,weight= weights_b)
def error_cal(test_data_o,test_data_b, k):
  test_o_result = []
  for i in range(len(test_data_o[:,0])):
    test_o_result.append(myknn(test_data_o[i],all_data = all_data,k = k))
  error_o = np.array(test_o_result) - np.zeros(shape = np.array(test_o_result).shape)

  test_b_result = []
  for i in range(len(test_data_b[:,0])):
    test_b_result.append(myknn(point = test_data_b[i],all_data=all_data,k=k))
  error_b = np.ones(shape=np.array(test_b_result).shape)-np.array(test_b_result)

  total_test_error = (error_o.sum() + error_b.sum())/400
  return( total_test_error)

error_cal(test_data_o = test_data_o,test_data_b=test_data_b,k=40)

error_plot = []
for i in range(100):
  error_plot.append(error_cal(test_data_o = test_data_o,test_data_b=test_data_b,k=i))

plt.plot(range(100),error_plot,label = "test_error")
plt.show()