import numpy as np

a = np.array([[1,2,3],
              [4,5,6]])

b = np.array([[1],
              [2]])

for (x, y) in zip(a, b):
    print((x, y))