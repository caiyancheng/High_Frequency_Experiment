import numpy as np

lms_gray = np.array([0.739876529525622, 0.320136241543338, 0.020793708751515])
mc1 = lms_gray[0] / lms_gray[1]
mc2 = (lms_gray[0] + lms_gray[1]) / lms_gray[2]

M_lms_dkl = np.array([[1,  1,    0],
                       [1, -mc1,  0],
                       [-1, -1,  mc2]])

inv_M = np.linalg.inv(M_lms_dkl)

for i, name in enumerate(["ach", "rg", "yv"]):
    v = inv_M[:, i]
    v_norm = v / np.sum(np.abs(v))   # L1 归一化
    print(f"{name}: {v_norm}")