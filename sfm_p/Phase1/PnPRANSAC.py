import numpy as np
from LinearPnP import *
from misc import *


def PnP_RANSAC(K, feature_points, X_points):
    """
    Perform the RANSAC algorithm using the
    :param K: camera intrinsic matrix of calibrated parameters
    :param feature_points: image plane feature points
    :param X_points: world points corresponding to the 2D image feature_points
    :return: the best C vector and R matrix of the new camera
    """

    iterations = 1000
    threshold = 1

    homogenized_world_points = get_homogenous_coordinates(X_points)
    u, v = feature_points.T
    max_inliers = []

    for iter in range(iterations):
        # sample 6 random points assuming they are inliers
        sample_inds = np.random.sample(range(feature_points.shape[0]-1), 6)
        sample_features = feature_points[sample_inds, :]
        sample_world_points = X_points[sample_inds, :]

        R, C = linear_PnP(K, sample_features, sample_world_points)

        T = -R @ C.reshape((3, 1))
        P = K @ np.concatenate((R,T), axis=1)
        P1, P2, P3 = P  # 4, 4, 4,

        u_num = np.dot(P1, homogenized_world_points.T)  # N,
        v_num = np.dot(P2, homogenized_world_points.T)  # N,
        denom = np.dot(P3, homogenized_world_points.T)  # N,

        u_ = u_num / denom
        v_ = v_num / denom

        error = (u - u_) ** 2 + (v - v_) ** 2

        # check if error is below some threshold for rest of the points
        curr_inliers = abs(error) < threshold # N,
        if np.sum(curr_inliers) > np.sum(max_inliers):
            max_inliers = curr_inliers

    feature_inliers = feature_points[max_inliers]
    world_point_inliers = X_points[max_inliers]

    # Re-estimate R and C for all the inliers
    R, C = linear_PnP(K, feature_inliers, world_point_inliers)

    return R, C
