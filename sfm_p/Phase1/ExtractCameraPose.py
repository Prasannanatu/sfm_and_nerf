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

    U, _, V_T = np.linalg.svd(E)                    # getting the value of U, V_T from the Essential Matrix.

    W = np.array([[0, -1, 0],                       # W matrix for computing rotation matrices
                  [1,  0, 0],
                  [0,  0, 1]])

    # Translation vector is the last column of U
    C_1 = U[:, 2]                                   # all rows of column 3
    C_2 = -U[:, 2]
    C_3 = C_1
    C_4 = C_2

    # Rotation matrix is E recomposed from SVD using the W matrix for the singular values D matrix
    R_1 = U @ W @ V_T

    # Enforce the orthonormal constraint which must hold for a valid rotation matrix by recomputing with SVD
    U_R_1, _, V_T_R_1 = np.linalg.svd(R_1)
    Identity = np.identity(3)                   # diagonal matrix of ones, orthonormal matrix
    R_1 = U_R_1 @ Identity @ V_T_R_1            # recompose the R matrix from the corrected SVD products

    R_2 = R_1

    R_3 = U @ W.T @ V_T                         # .T is equivalent to np.transpose()

    U_R_3, _, V_T_R_3 = np.linalg.svd(R_3)
    R_3 = U_R_3 @ Identity @ V_T_R_3            # recompose the R matrix from the corrected SVD products

    R_4 = R_3

    # Form the lists to return
    C_list = [C_1, C_2, C_3, C_4]
    R_list = [R_1, R_2, R_3, R_4]

    # Enforcing right-hand coordinate system, determinant of rotation matrices must be 1 not -1
    for i, R in enumerate(R_list):
        det_R = np.linalg.det(R)
        # print('Det of R' + str(i) + ': ' + str(det_R))
        if det_R < 0:
            R_list[i] = -R                      # invert the values
            C_list[i] = -C_list[i]

    return C_list, R_list
