import numpy as np

# service desk: 0, anesthetist: 1, CPM: 2, Additional tests: 3, home: 4
P = np.array([
    [0, 0.75, 0.25, 0, 0],
    [0, 0, 0.21, 0.055, 0.735],
    [0, 1/3, 0, 1/3, 1/3],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0],
])

gamma = np.array([8.3, 0, 0, 0, 0])

# lam = gam + lam*P
# lam(lam*P) = gam
# lam = gam*(I-P)^-1
lam = gamma @ np.linalg.inv((np.eye(5) - P))
print(lam)