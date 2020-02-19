import numpy as np

a = np.arange(9).reshape(3, 3)

b = np.arange(10,19).reshape(3, 3)

print(a.type)
print(b)
c = np.argwhere(a != b)
print(len(c))