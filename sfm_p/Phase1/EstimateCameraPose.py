import numpy as np
import math
import random
import matplotlib.pyplot as plt



def estimate_camera_pose(K, E):
    """
    Input: Essential matrix 
    getting the value of U, V_T from the Essential Matrix.
    



    Output: 4 possible Camera Poses.

    

    """
    R= []
    P =[]
    T =[]
    U,D,V_T = np.linalg.svd(E)

    W = np.array[[0, -1, 0],
                 [1, 0, 0],
                 [0, 0, 1]]
    C= []
    C1 = U[:, 3]
    C2 = -U[:,3]
    C.append(C1)
    C.append(C2)
    

    R_1 = U@W@V_T
    R.append(R_1)
    R_2 = U@(np.transpose(W))@V_T
    R.append(R_2)
    ones = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    
    R_n = []
    T_n = []
    C_n = []
    for i in range(len(R)):
        for j in range(len(C)):

            if (np.linalg.det(R[i] < 0)):
                
                R[i] = -R[i]
                C[j] = -C[j]

            T = np.hstack(ones ,-C[j])
            P.append(K@R@T)
            R_n.append(R)
            T_n.append(T)
            C_n.append(C)






    # P_1 =  K @ R_1  @ Translation_1
    # P_2 =  K @ R_1  @ Translation_2
    # P_3 =  K @ R_2  @ Translation_1
    # P_4 =  K @ R_2  @ Translation_2

    # return P_1, P_2, P_3, P_4




    return R_n, T_n, P,C_n


