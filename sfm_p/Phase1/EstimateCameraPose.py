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
    
    C1 = U[:, 3]
    C2 = -U[:,3]
    

    R_1 = U@W@V_T
    R.append(R_1)
    R_2 = U@(np.transpose(W))@V_T
    R.append(R_2)
    ones = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    Translation_1 = np.hstack(ones ,-C1)
    Translation_2 = np.hstack(ones, -C2)
    R_n = []
    T_n = []
    T.append(Translation_1)
    T.append(Translation_2)
    dict ={}
    for i in range(len(R)):
        for j in range(len(T)):
            if (np.linalg.det(R[i] < 0)):
                R[i] = -R[i]
                T[i] = -T[i]
            P.append(K@R@T)
            R_n.append(R)
            T_n.append(T)






    # P_1 =  K @ R_1  @ Translation_1
    # P_2 =  K @ R_1  @ Translation_2
    # P_3 =  K @ R_2  @ Translation_1
    # P_4 =  K @ R_2  @ Translation_2

    # return P_1, P_2, P_3, P_4




    return R_n, T_n, P


