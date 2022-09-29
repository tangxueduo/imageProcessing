import numpy as np

a = np.zeros((3, 4, 4))
b = np.ones((3, 4, 4))

a[1,:,:] = 2
b[0,0,:] = 0
tmp = np.where(((a>0)& (b>0) & (a<b)))
print(tmp)
i = 1