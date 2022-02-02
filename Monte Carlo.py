import numpy as np
def lin(x):
    return(x)


def mc(n):
    k = 0
    x = np.random.uniform(0,1,n)
    y = np.random.uniform(0,1,n)# make the point with y for each x point
    for i in range(n):
        if y[i] < lin(x[i]): ##for x = x if y(x)< f(x) then the point is under the line use all point/1 = the point under the line/
integral area
            k = k+1
    k = np.array(k)
    return k/n

mc(1000000)