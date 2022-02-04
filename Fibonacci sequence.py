data_f = [1,1]
def f_function(n):
    for i in range(n):
        data_f.append(data_f[i+1] + data_f[i])
    return(data_f)

fi = np.array(f_function(100))
fi.shape
