import numpy as np
from misc import *
from scipy.optimize import least_squares



def get_reoprojection_error_total(vec1, vec2, X, P, K):
    """
    Inputs: vector1 and vector 2  are the two feature_match coordinates of two respective images
            
    
    """
    error = []
    ones = np.identity(3)
    R_0 = np.identity(3)
    C_0 = np.zeros(3,3)
    T_0 = np.hstack(ones, C_0)
    P_0 = K @ R_0 @ T_0
    Xw_optimised = []
    for i in range(vec1.shape[0]):
        error_1 = reprojection_error(vec1[i],X[i], P_0)
        error_2 = reprojection_error(vec2[i],X[i],P) 
        error= [error_1, error_2]
        output = least_squares(error, x0 = X[i],method = 'lm', kwargs={vec1, vec2, P_0, P})
        Xw_optimised.append(output.x)

    Xw_optimised = np.vstack(Xw_optimised)
    Xw_optimised = Xw_optimised/Xw_optimised[3]
    Xw_optimised = get_unhomogenous_coordinates(Xw_optimised)
    return Xw_optimised





    return 0

def reprojection_error(v,X,P):
    """
    Its a geometric error between the actual image point and reprojected points from the world
    Inputs: The actual image coordinates and reporjected points

    Outputs: The error between those two points
    """ 
    #Getting the reporjected points of the image
    v_reproj= X.T @ P
    v_reproj = v/v[2]
    reprojected_error = (v-v_reproj)**2
    reprojected_error =  reprojected_error[0:2]

    return reprojected_error


