from k_means import cross_euclidean_distance
import pandas as pd
import numpy as np

data = np.array(([1,2], [1,3], [1, 2]))
print(data)
df = pd.DataFrame(data, columns = ['x0', 'x1'])
a = np.array([1,0,1])
b = a == 1
print(b)
print(np.average(df.iloc[b], axis = 0))
a = np.random.rand(1,3)
print(a)
a = cross_euclidean_distance(df.to_numpy(), np.array([[1,2], [1,3]]))
print(np.argmin(a, axis = 1))
print(df.iloc[np.array([True, False, True])].to_numpy())
print(np.sqrt(np.power(np.array([1.2,1.2,.2]),2)))