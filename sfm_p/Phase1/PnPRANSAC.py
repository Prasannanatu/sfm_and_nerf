import numpy as np
import random
from LinearPnP import *
from misc import *


def PnP_RANSAC(K, feature_points, X_points):
    """
    Perform the RANSAC algorithm using linear PnP
    :param K: camera intrinsic matrix of calibrated parameters
    :param feature_points: image plane feature points
    :param X_points: world points corresponding to the 2D image feature_points
    :return: the best C vector and R matrix of the new camera
    """

    iterations = 5000
    threshold = 10

    num_points = feature_points.shape[0]

    # world_points_homogen = get_homogenous_coordinates(X_points)
    world_points_homogen = X_points

    u = feature_points[:, 0]
    v = feature_points[:, 1]
    # print('feature points:', feature_points)

    max_inliers = []

    for index in range(iterations):                                        # index iterator variable non-use intentional

        # Select 6 points at random
        random_indeces = random.sample(range(num_points - 1), 6)

        sample_features = feature_points[random_indeces, :]
        sample_world_points = X_points[random_indeces, :]

        # Perform linear PnP to calculate the pose of the camera
        R, C = linear_PnP(K, sample_features, sample_world_points)

        C = C.reshape((3, 1))

        T = -R @ C
        R_T = np.concatenate((R, T), axis=1)
        P = K @ R_T
        P_1, P_2, P_3 = P

        u_numerator = np.dot(P_1, world_points_homogen.T)
        v_numerator = np.dot(P_2, world_points_homogen.T)

        denominator = np.dot(P_3, world_points_homogen.T)

        u_ = u_numerator / denominator
        v_ = v_numerator / denominator

        error = np.sqrt( (u - u_) ** 2 + (v - v_) ** 2 )

        current_inliers = abs(error) < threshold
        inliers_sum = np.sum(current_inliers)
        max_inliers_sum = np.sum(max_inliers)

        if inliers_sum > max_inliers_sum:
            max_inliers = current_inliers

    feature_inliers = feature_points[max_inliers]
    world_point_inliers = X_points[max_inliers]

    # Re-estimate R and C for all the inliers
    R, C = linear_PnP(K, feature_inliers, world_point_inliers)

    return R, C
