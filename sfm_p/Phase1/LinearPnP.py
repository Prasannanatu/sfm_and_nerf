import numpy as np


def linear_PnP(K, feature_points, X_points):
    """
    Calculate the pose of a new camera capturing an additional image of the scene
    :param K: camera intrinsic matrix of calibrated parameters
    :param feature_points: image plane feature points
    :param X_points: world points corresponding to the 2D image feature_points
    :return: the C vector and R matrix of the new camera
    """

    u = feature_points[:, 0]
    v = feature_points[:, 1]

    X = X_points[:, 0]
    Y = X_points[:, 1]
    Z = X_points[:, 2]

    zeros = np.zeros_like(X)          # vector of ones the length of X_points
    ones = np.ones_like(X)            # vector of zeros the length of X_points

    # Formulate the system of equations to solve using SVD
    # DLT (Direct Linear Transformation) from Unsupervised Deep Homography paper Ty Nguyen et. all
    A1 = np.vstack([X, Y, Z, ones, zeros, zeros, zeros, zeros, -u * X, -u * Y, -u * Z, -u]).T
    A2 = np.vstack([zeros, zeros, zeros, zeros, X, Y, Z, ones, -v * X, -v * Y, -v * Z, -v]).T
    A = np.vstack([A1, A2])

    # Perform SVD on the system of equations
    U, D, V_T = np.linalg.svd(A)                        # output matrix V_T is shape [12 x 12]
    P = V_T[-1, :]                                      # -1 refers to the last element, the maximum index number
    P = P.reshape((3, 4))                               # reshape to form the projection matrix P [3 x 4]

    # Compute the rotation matrix R
    R_camera = P[0:3, 0:3]
    K_inv = np.linalg.inv(K)
    R = K_inv @ R_camera

    U_R, D_R, V_T_R = np.linalg.svd(R)                  # output matrix V_T_R is shape [3 x 3]
    R = U_R @ V_T_R                                     # enforcing orthogonality

    lamda = D_R[0]                                      # scale factor is first singular value

    t = P[:, 3]
    T = K_inv @ t / lamda

    # Enforcing right-hand coordinate system, determinant of rotation matrices must be 1 not -1
    R_det = np.linalg.det(R)

    if R_det < 0:
        R = -R
        T = -T

    C = -R.T @ T

    return R, C
