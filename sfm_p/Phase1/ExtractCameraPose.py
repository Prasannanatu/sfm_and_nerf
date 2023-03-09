import numpy as np
import math
import random
import matplotlib.pyplot as plt


def extract_camera_pose(E):
    """
    Calculate the possible camera poses in a scene using the essential matrix
    :param E: the camera essential matrix
    Output: list of 4 possible camera center matrices C and rotation matrices R
    """

    U, _, V_T = np.linalg.svd(E)                # getting the value of U, V_T from the Essential Matrix.

    W = np.array([[0, -1, 0],                   # W matrix for computing rotation matrices
                  [1,  0, 0],
                  [0,  0, 1]])

    C_1 = U[:, 2]                               # all rows of column 3
    C_2 = -U[:, 2]
    C_3 = C_1
    C_4 = C_2

    R_1 = U @ W @ V_T
    R_2 = R_1
    R_3 = U @ W.T @ V_T                         # .T is equivalent to np.transpose()
    R_4 = R_3

    C_list = [C_1, C_2, C_3, C_4]
    R_list = [R_1, R_2, R_3, R_4]

    # Enforcing right-hand coordinate system, determinant of rotation matrices must be 1 not -1
    for i, R in enumerate(R_list):
        det_R = np.linalg.det(R)
        print('Det of R' + str(i) + ': ' + str(det_R))
        if det_R < 0:
            R_list[i] = -R                      # invert the values
            C_list[i] = -C_list[i]

    return C_list, R_list

    # R = []
    # P = []
    # T = []
    #
    # U, D, V_T = np.linalg.svd(E)              # getting the value of U, V_T from the Essential Matrix.
    #
    # W = np.array([[0, -1, 0],
    #              [1, 0, 0],
    #              [0, 0, 1]])
    # C= []
    # C1 = U[:, 2]
    # C2 = -U[:,2]
    # C.append(C1)
    # C.append(C2)
    #
    #
    # R_1 = U@W@V_T
    # R.append(R_1)
    # R_2 = U@(np.transpose(W))@V_T
    # R.append(R_2)
    # ones = np.array([[1, 0, 0],
    #                 [0, 1, 0],
    #                 [0, 0, 1]])
    #
    # R_n = []
    # T_n = []
    # C_n = []
    # for i in range(len(R)):
    #     for j in range(len(C)):
    #
    #         if (np.linalg.det(R[i] < 0)):
    #
    #             R[i] = -R[i]
    #             C[j] = -C[j]
    #         c = np.asarray(C[j])
    #         c = c.reshape((3,1))
    #         T = -R[i] @ c
    #         T_t = np.hstack((R[i] ,T))
    #         P.append(K @ T_t)
    #         # P.append(K@R@T)
    #         R_n.append(R[i])
    #         T_n.append(T_t)
    #         C_n.append(C)



    # P_1 =  K @ R_1  @ Translation_1
    # P_2 =  K @ R_1  @ Translation_2
    # P_3 =  K @ R_2  @ Translation_1
    # P_4 =  K @ R_2  @ Translation_2

    # return P_1, P_2, P_3, P_4




    # return R_n, T_n, P,C_n

