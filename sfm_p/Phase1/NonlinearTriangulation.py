import numpy as np
from misc import *
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('tkagg')


def non_linear_triangulation(K, C, R, best_matched_points, X_points):
    """
    Refine the world points triangulated from two image views by minimizing the re-projection error
    using scipy.optimize.least_squares()
    :param K: camera intrinsic matrix of calibrated parameters
    :param C: camera origin position vector also known as T
    :param R: camera orientation expressed as a rotation matrix
    :param best_matched_points: filtered best matched image plane points between the two views
    :param X_points: world points triangulated using linear_triangulation()
    :return: the list of refined world points
    """

    num_points = len(X_points)
    p1 = best_matched_points[:, 0]
    p2 = best_matched_points[:, 1]
    # Convert to homogenous coordinates
    X_points = get_homogenous_coordinates(X_points)

    # Calculate the camera pose from the position vector and rotation matrix
    C = C.reshape((3, 1))                               # make into column vector
    Identity = np.identity(3)                           # identity matrix
    I_C = np.append(Identity, -C, axis=1)               # [3 x 4] matrix

    P = K @ R @ I_C

    # Calculate the pose for the camera at the origin, the assumed location of the camera that captured u_v_1
    R_O = Identity
    C_O = np.zeros((3, 1))
    I_C_O = np.append(Identity, C_O, axis=1)
    P_O = K @ R_O @ I_C_O

    X_points_nonlin = []
    print(f"the best matches point{best_matched_points}")
    print(f"the best matches point{best_matched_points[0]}")
    print(f"the best matches point{best_matched_points[1]}")

    # Optimize every point in X_points to get the list of refined points x_points_nonlin
    for i in range(num_points):

        X = X_points[i]                                 # current point used as initial guess

        # X_opt = least_squares(error_function, x0=X, method='trf',
        #                       args=[p1,p2 , P, P_O])
        X_opt = least_squares(lambda x: error_function(x, p1,p2, P, P_O, i), x0=X, method='trf')


        X_opt = X_opt.x/X_opt.x[-1]     # divide by last value homoginize

        X_points_nonlin.append(X_opt)

    return X_points_nonlin


def error_function(x, p1,p2, P, P_0, point_index):
    """
    Error function to be used in scipy.optimize.least_squares()
    :param x: the vector of parameters to find through the optimization, initial guess is known as x0
    :param best_matched_points: filtered best matched image plane points between the two views
    :param P: computed projection matrix of the second image
    :param P_0: computed projection matrix of the first image at the origin
    :param point_index: the index of the current point in best_matches_points
    :return: the vector of parameter errors for recomputing in scipy.optimize.least_squares()
    """

    # best_matched_points = np.asarray(best_matched_points)
    # u_v_1 = best_matched_points[:, 0]
    # u_v_2 = best_matched_points[:, 1]

    point_1 = p1[point_index]
    point_2 = p2[point_index]

    error_reproj_1 = get_reprojection_error(point_1, x, P_0)
    error_reproj_2 = get_reprojection_error(point_2, x, P)

    error = np.concatenate((error_reproj_1, error_reproj_2))

    return error


def get_reprojection_error(point_image, point_world, P):
    """
    Geometric error between the actual image point and reprojected points from the world
    :param point_image: image point coordinates (u, v) or (x, y)
    :param point_world: world point coordinates [x, y, z]
    :param P: projection matrix to transform between point_image and point_world
    :return: the re-projection error of (P * point_world) - point_image
    """

    # [x, y] to [x, y, 1] for homogeneous coordinates
    point_image = np.array([point_image[0], point_image[1], 1], np.float32)

    point_reproj = P @ point_world.T                    # getting the reporjected points of the image

    point_reproj = point_reproj / point_reproj[2]       # must divide x and y by z to get 2D points again

    error_reproj = (point_reproj - point_image) ** 2

    error_reproj = error_reproj[0:2]

    return error_reproj


