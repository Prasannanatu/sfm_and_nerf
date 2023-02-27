import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import random


def get_Essential_Matrix(F, K):
    """
    Input: fundamental Matrix and the Intrinsic Matrix for the Image.

    Output: Essential matrix which can be used to calculate the camera poses.

    We Know that : E  = K^T. F. K

    """
    E = K.T @ F @ K
    # The essential matrix may have ambiguity and noise and hence trying to get rid of the noise using svd
    #making diagonal matrix D as [1, 1, 0]

    U,D,V_T = np.linalg.svd(E)

    D_new = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0 ,0]])
    
    E = U @ D_new @ V_T

    return E


