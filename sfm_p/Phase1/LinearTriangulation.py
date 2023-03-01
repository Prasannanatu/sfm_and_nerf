import numpy as np
from misc import *
#import pry



def linearTriangulation(R_n,T_n,P,K, vec1,vec2):

    """
    Inputs: camera poses(C,R)
            camera Intrinsic matrix  as K,
            two image coordinates from the two images (vec1, vec2)

    Output:
            Getting the 3D values for the points

    """
    # Converting the images unhomogenous coordinates to homogenous Coordinates
    vec1_ = [get_homogenous_coordinates(vec1_val) for vec1_val in vec1]
    vec2_ = [get_homogenous_coordinates(vec2_val) for vec2_val in vec2]

    # getting the rotation and Translation matrix for origin camera Pose.
    ones = np.identity(3)
    R_0 = np.identity(3)
    C_0 = np.zeros((3,3))
    T_0 = -R_0 @ C_0
    # T_0 = np.hstack((R_0, T_0))
    P_0 = K @ np.hstack((R_0, T_0))
    X =[]


    for i in range(len(R_n)):
        for j in range(len(vec1_)):
            #pry()
            X_1 = skew_matrix(vec1[j]) @ P_0
            
            X_2 = skew_matrix(vec2[j]) @ P[i]
        
            #pry()
            X_ = np.vstack((X_1, X_2))

            U, D, V_T = np.linalg.svd(X_)
            X_w = V_T[-1]
            X_w = X_w/X_w[-1]
            X.append(X_w)
        X = np.array(X)
        X_final = np.hstack(X_final, X)
    return X_final

