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
    # Getting the last row for set of point correspondances as zero.
    vec_one = np.ones(vec1.shape[0])

    # Getting the respective coordinates from the vector.
    x_1, y_1, x_2, y_2 = vec1[:, 0], vec1[:, 1], vec2[:, 0], vec2[:, 1]

    # Getting the A matrix from the provided equation for SVD formulation.
    # A = np.asarray([x_1 * x_2, x_1 * y_2, x_1, y_1 * x_2, y_1 * y_2, y_1, x_2, y_2, vec_one])
    A = np.asarray([x_1 * x_2, y_1 * x_2, x_2, x_1 * y_2, y_1 * y_2, y_2, x_1, y_1, vec_one])  # N x 9
    A = np.transpose(A)                                 # transpose so matrix is [m x 9], m = matched point pairs

    # Perform SVD
    _, _, V_T = np.linalg.svd(A)                        # output matrix V_T is shape [9 x 9]
    f = V_T[-1, :]                                      # -1 refers to the last element, the maximum index number

    f = f.reshape(3, 3)                                 # reshape to [3 x 3] matrix

    # Re-estimate F by performing SVD on the F calculated from the matched points
    U_F, D_F, V_T_F = np.linalg.svd(f)

    # Enforcing the internal constraint which the computed F matrix must satisfy, must be of rank 2
    # The last element of computed D_F (singular values) should be 0 but due to noise may instead be close to 0
    D_F[2] = 0.0                                        # make D_F rank 2 by making the third singular value 0
    diag_D_F = np.diag(D_F)                             # diagonal matrix of the singular values D_F
    F = U_F @ diag_D_F @ V_T_F                          # recompose the F matrix from the corrected SVD products

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
 








 

