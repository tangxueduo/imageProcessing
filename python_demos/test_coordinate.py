import numpy as np
X = np.array([
    [-1,0,0],
    [0,1,0],
    [0,0,1]
])
Y = np.array([
    [-1, 0,0],
    [0, 0, 1],
    [0, 1, 0],
])
Z = np.array([
    [0, 0, -1],
    [-1, 0, 0], 
    [0, 1, 0], 
])

M1 = np.array([
    [0.367397,0, -0.930064],
    [-0.930064, 0, -0.367397],
    [0, 1, 0]
])

# M = np.dot(np.dot(X,Y), Z)
# M = np.dot(Y,M1)
M =M1
print(M)

st_origin = [-127.2509765625, -303.7509765625, -713.2999877929688]
ed_origin = [ 127.74902344,  -48.75097656, -532.29998779]
new_st_origin = np.dot(st_origin, M).tolist()
new_ed_origin = np.dot(ed_origin, M).tolist()
print(f'new_st_origin: {new_st_origin}, new_ed_origin: {new_ed_origin}')
