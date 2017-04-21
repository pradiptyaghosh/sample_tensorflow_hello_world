import numpy as np 
chunk_size = 4
n_chunk = 5
x = np.ones((1,2,20))
print(x)
x = np.transpose(x, [1,0,2])
print('transform:', x)
x = np.reshape(x, [-1, chunk_size])
print('reshape:', x)
x = np.split(x,n_chunk,0)
print('split:', x)