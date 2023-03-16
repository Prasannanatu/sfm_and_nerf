import numpy as np
from misc import *
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def error_function(x, K, feature_points, X_points):

    Q = x[0:4]
    #C = x[4:].reshape(-1, 1)                   # -1 means use the maximum size

    R = Rotation.from_quat(Q).as_matrix()
    R = R.reshape((3, 3))

    # Calculate the camera pose from the position vector and rotation matrix
    #C = C.reshape((3, 1))
    t = x[4:7]
    t = t.reshape((3, 1))

    P = np.hstack([R, t])

    errors = []

    for X, point in zip(X_points, feature_points):

        u = point[0]
        v = point[1]

        p_1_T, p_2_T, p_3_T = P                        # getting the rows of P

        p_1_T = p_1_T.reshape(1, -1)
        p_2_T = p_2_T.reshape(1, -1)
        p_3_T = p_3_T.reshape(1, -1)

        X = get_homogenous_coordinates(X.reshape(1, -1)).reshape(-1, 1)

        u_projected = np.divide(p_1_T.dot(X), p_3_T.dot(X))
        v_projected = np.divide(p_2_T.dot(X), p_3_T.dot(X))

        error = (v - v_projected) ** 2 + (u - u_projected) ** 2

        errors.append(error)

    error_total = np.mean(np.array(error).squeeze())

    return error_total


def nonlinear_PnP(K, feature_points, X_points, R, C):

    Q = Rotation.from_matrix(R).as_quat()

    X = [Q[0], Q[1], Q[2], Q[3], C[0], C[1], C[2]]

    optimized_params = least_squares(fun=error_function, x0=X, method="trf", args=[X_points, feature_points, K])

    X_opt = optimized_params.x

    Q = X_opt[:4]
    C = X_opt[4:]
    R = Rotation.from_quat(Q).as_matrix()

    return R, C
