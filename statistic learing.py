import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

auto = pd.read_csv('/Users/leehao-mbp/Desktop/auto-mpg.csv',header=None)
auto = auto.drop([8],axis=1)
auto = auto.drop([0],axis=0)
auto = auto.dropna()###删除所有有NA的列
auto = auto.drop(auto[auto[3] == "?"].index)
np.array(auto[5]).reshape(398,1)
auto1 = auto.to_numpy()
auto = auto1.astype(float)
np.mean(auto[:,7])
auto[:,7]
y = [0 if x <= np.mean(auto[:,5]) else 1 for x in auto[:,5]]
auto_all = np.c_[auto,y]
auto_all
x_train, x_test, y_train, y_test = train_test_split(auto,y,test_size=0.3)
###svm
model_svm = SVC()
model_svm.fit(x_train,y_train)
np.sum(abs(model_svm.predict(x_test) - np.array(y_test)))/len(x_test[:,1])


##lda
model_lad = LinearDiscriminantAnalysis(store_covariance = True) ##需要单独计算协方差矩阵
model_lad.fit(x_train,y_train)
y_test_lda = model_lad.predict(x_test)
np.sum(np.abs(y_test_lda - y_test))/len(y_test)


##qda
model_qda = QuadraticDiscriminantAnalysis(store_covariance = True)
model_qda.fit(x_train,y_train)
np.sum(np.abs(model_qda.predict(x_test) - y_test))/len(y_test)


#random forest
nt = np.array([1,2,3,4,5,6,10,100,1000])
nt_error = []
for i in range(len(nt)):
    rf = RandomForestClassifier(n_estimators = nt[2])##几棵树
    rf.fit(x_train,y_train)
    nt_error.append(np.sum(np.abs(rf.predict(x_test) - y_test))/len(y_test))
rf = RandomForestClassifier(n_estimators = nt[2])##几棵树
rf.fit(x_train,y_train)
np.sum(np.abs(rf.predict(x_test) - y_test))/len(y_test)

##logist
logist = LogisticRegression()
logist.fit(x_train,y_train)
logist.perdict(x_test)


##knn
knn = neighbors.KNeighborsClassifier(2)
knn.fit(x_train,y_train)
sum(abs(knn.predict(x_test) - y_test))/len(y_test)

