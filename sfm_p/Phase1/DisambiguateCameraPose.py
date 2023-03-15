import numpy as np
import matplotlib.pyplot as plt
import random
import math


def disambiguate_camera_poses(C_list, R_list, X_points_poses):
    """
    Disambiguate between the camera poses in the lists to find which pose has the most points in front of the camera
    :param C_list: list of possible camera center translation vectors C
    :param R_list: list of possible camera rotation matrices R
    :param X_points_poses: list of triangulated points for a camera pose C and R pair
    :return: The correct C from C_list, R from R_list, and X_points from X_points_poses
    """

    num_poses = len(R_list)

    total_count = []                # create an empty list for getting the count of true values for all possible poses

    for i in range(num_poses):                              # looping on 4 possible X, C, R values

        R = R_list[i]                                       # getting the current R value
        C = C_list[i]                                       # getting the current C value
        X_points = X_points_poses[i]                        # getting the current X value

        r_3 = R[:, 2]                                       # getting the current r3 value

        count = 0                                           # initializing a counter
        num_points = len(X_points)

        for j in range(num_points):                         # looping over all triangulated points to check cheirality

            X = X_points[j]                                 # current point
            z = X[2]                                        # points are in the form [x, y, z]

            cond_1 = r_3.T @ (X.T - C) > 0                  # The cheirality condition
            cond_2 = z > 0                                  # is Z point positive (in front of image plane)

            if cond_1 and cond_2:

                count = count + 1                           # if yes consider it

        total_count.append(count)                           # append all values to a list after checking all points

    # print('total count: ', total_count)
    total_count = np.array(total_count)                     # converting the list to array.
    idx = np.argmax(total_count)                            # get the max idx of the all.

    R_correct = R_list[idx]                                 # get the final value of R
    C_correct = C_list[idx]                                 # get the final value of Camera Pose
    X_correct = X_points_poses[idx]                         # get the final coordinates of the World Points

    return C_correct, R_correct, X_correct, idx






    

        
        
