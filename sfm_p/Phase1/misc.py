import numpy as np
import math
import random
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix


def skew_matrix(x):
    X = np.array([[0, -x[2] , x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
    return X


def get_homogenous_coordinates(coordinates):
    """
    Input: The co-ordinates u,v
    Outputs : The homogenize coordinates
    """

    coordinates = np.asarray(coordinates)
    ones = np.ones((coordinates.shape[0], 1))

    homo = np.concatenate((coordinates, ones), axis=1)

    return homo


def get_unhomogenous_coordinates(coordinates):
    """
    
    """
    coordinates = np.asarray(coordinates)

    unhomo = np.delete(coordinates, coordinates.shape[1]-1, axis =1)

    return unhomo


def get_K():
    path = '../Data/calibration.txt'
    K = np.empty((3,3))
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line_numbers = line.strip().split()  # extract the float numbers from the line
            for j, num in enumerate(line_numbers):
                line_number = float(num)
                K[i][j] = line_number
            # line_numbers = [float(num) for num in line_numbers]  # convert the numbers to float
            # K[]

    # k =np.asarray(K)
    # k.reshape((3,3))
    return K


def find_matching_points(world_points_1, matched_points_1_2, matched_points_1_3, u_v_1_3):
    """
    Use the matches between points_1_2 and points_1_3 to find the corresponding
    matching points between 2 and 3 which can be related through 1
    Also matches them to the world points passed in
    """

    world_points_1 = np.asarray(world_points_1)
    # print('wld pt shape:', world_points_1.shape)
    # world_points_1.reshape((-1, 1, 4))

    u_v_1_12 = matched_points_1_2[:, 0]
    # print('u_v_1_2 shape:', u_v_1_12.shape)
    u_v_1_13 = matched_points_1_3[:, 0]

    mask = np.isin(u_v_1_12, u_v_1_13)
    indices = np.where(mask == 1)[0]        # the indices of points in 1_2 which also exist in 1_3
    indices = np.unique(indices)            # returned in pairs because a point has two value, remove the other

    x = u_v_1_12[indices]                   # get those points from 1_2 using the indices

    mask_ = np.isin(u_v_1_13, x)
    indices_1 = np.where(mask_ == 1)[0]      # the indices of 1_3 which also exist in 1_2
    indices_1 = np.unique(indices_1)

    # uv_13 = u_v_1_13[indices_1]
    uv_13 = u_v_1_3[indices_1]              # get the final image point correspondences
    # print("mached points_13: ", uv_13)

    world_points_1_3 = world_points_1[indices_1]

    return uv_13, world_points_1_3


def get_Rotation(Q, type_='q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()

    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()


def Rotation_on_euler(R):
    euller = Rotation.from_matrix(R)

    return euller.as_rotvec()


