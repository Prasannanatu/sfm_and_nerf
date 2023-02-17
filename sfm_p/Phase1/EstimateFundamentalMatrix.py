import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


def get_fundamental_matrix(vec1, vec2):
    """Input for the function will be two vectors for the same point in two images
    
    Input : vec1, vec2  point correspondances of the two images.
    Output : Fundamental matrix of shape 3*3 which gives us the relation between two relative points.
    i.e. x_2.T * F * x_1.T  = 0 
    """
    #Getting the last row for set of point correspondances as zero.
    vec_one = np.ones(vec1.shape[0])

    #Getting the respective coordinates from the vector.
    x_1, y_1, x_2, y_2 = vec1[:,0], vec1[:,1], vec2[:,0], vec2[:,1]

    #getting the A matrix from the provided equation for SVD formulation.
    A = [x_1 * x_2, x_1 * y_2, x_1, y_1 * x_2, y_1 * y_2, y_1, x_2, y_2, vec_one]

    #singular Value Decomposition
    U,D,V_T = np.linalg.svd(A)

    x = V_T[:,-1]
    F = x.reshape(3,3)
    F = np.transpose(F)
    # Enforcing rank 2 on the matrix as there are only eight equations and eight unknown
    #however due to noise in the Image this doesn't turn out to be zero.
    #Making the last column last element zero changes the rank from  3 to 2.
    F[2,2] = 0

    return F


def get_epipolar_points(F):
    """ 
    get the epipolar points for the matches
    
    Input for this will be the fundamental matrix F calculated above

    Output: Epipole for the corresponding lines
    """

    U,D,V_T = np.linalg.svd(F)

    epipole_1 = V_T[:,2]
    epipole_2 = U[:,2]

    return epipole_1, epipole_2



def get_epipolar_lines(F, vec_1, vec_2):
    """
    This gives the output epipolar lines
    Input:
    Both the points correspondances abd the Fundamental Matrix

    Output:
    Line corresponding to the correspondances of other image
    """

    
    epipolar_line_1 = np.matmul(np.transpose(F), np.transpose(vec_2))
    epipolar_line_2 = np.matmul(F, np.transpose(vec_1))

    return epipolar_line_1, epipolar_line_2
 








 

