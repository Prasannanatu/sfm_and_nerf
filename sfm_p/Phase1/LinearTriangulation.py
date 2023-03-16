import matplotlib.pyplot as plt
import numpy as np
from misc import *
#import pry
from scipy.spatial.transform import Rotation


def linear_triangulation(K, C, R, best_matched_points):
    """
    Perform triangulation using linear least squares on the
    :param K: camera intrinsic matrix of calibrated parameters
    :param C: camera origin position vector also known as T
    :param R: camera orientation expressed as a rotation matrix
    :param best_matched_points: filtered best matched image plane points between the two views
    :return: the 3D points w.r.t. the camera origin and without a scale factor thus dimensionless
    """

    u_v_1 = best_matched_points[:, 0]
    u_v_2 = best_matched_points[:, 1]

    ones = np.ones((u_v_1.shape[0], 1))

    # Convert the image coordinates to homogenous coordinates
    points_1 = np.concatenate((u_v_1, ones), axis=1)
    points_2 = np.concatenate((u_v_2, ones), axis=1)
    # print(points_1)

    # Calculate the camera pose from the position vector and rotation matrix
    C = C.reshape((3, 1))                           # make into column vector
    Identity = np.identity(3)                       # identity matrix
    I_C = np.append(Identity, -C, axis=1)           # [3 x 4] matrix

    P = K @ R @ I_C
    # print('P: ', P)

    # Calculate the pose for the camera at the origin, the assumed location of the camera that captured u_v_1
    R_O = Identity
    C_O = np.zeros((3, 1))
    I_C_O = np.append(Identity, C_O, axis=1)
    P_O = K @ R_O @ I_C_O

    X_pts = []
    num_points = best_matched_points.shape[0]

    for i in range(num_points):

        X_1_i = skew_matrix(points_1[i]) @ P_O
        X_2_i = skew_matrix(points_2[i]) @ P

        x_P = np.vstack((X_1_i, X_2_i))
        _, _, V_T = np.linalg.svd(x_P)
        X_pt = V_T[-1][:]

        X_pt = X_pt / X_pt[3]
        # print('X_pt: ', X_pt)

        X_pt = X_pt[0:3]                    # discard the fourth value to make non homogeneous coordinates

        X_pts.append(X_pt)

    return X_pts


def visualize_points_2D(points_list_1, points_list_2, points_list_3, points_list_4):

    points_list_1 = np.asarray(points_list_1)               # use numpy to efficiently get all rows of a column
    points_list_2 = np.asarray(points_list_2)
    points_list_3 = np.asarray(points_list_3)
    points_list_4 = np.asarray(points_list_4)

    x_pts_1, y_pts_1, z_pts_1 = points_list_1[:, 0], points_list_1[:, 1], points_list_1[:, 2]
    x_pts_2, y_pts_2, z_pts_2 = points_list_2[:, 0], points_list_2[:, 1], points_list_2[:, 2]
    x_pts_3, y_pts_3, z_pts_3 = points_list_3[:, 0], points_list_3[:, 1], points_list_3[:, 2]
    x_pts_4, y_pts_4, z_pts_4 = points_list_4[:, 0], points_list_4[:, 1], points_list_4[:, 2]

    # Creating plot
    # fig = plt.Figure()
    dot_size = 1
    axes_lim = 30

    plt.scatter(x_pts_1, z_pts_1, color="red", s=dot_size)
    plt.scatter(x_pts_2, z_pts_2, color="blue", s=dot_size)
    plt.scatter(x_pts_3, z_pts_3, color="green", s=dot_size)
    plt.scatter(x_pts_4, z_pts_4, color="purple", s=dot_size)

    plt.title("triangulated world points")
    plt.xlim(-axes_lim, axes_lim)
    plt.ylim(-axes_lim, axes_lim)
    plt.xlabel("x (dimensionless)")
    plt.ylabel("z (dimensionless)")

    # show plot
    plt.show()


def visualize_points_lin_nonlin(points_list_1, points_list_2):

    points_list_1 = np.asarray(points_list_1)               # use numpy to efficiently get all rows of a column
    points_list_2 = np.asarray(points_list_2)

    x_pts_1, y_pts_1, z_pts_1 = points_list_1[:, 0], points_list_1[:, 1], points_list_1[:, 2]
    x_pts_2, y_pts_2, z_pts_2 = points_list_2[:, 0], points_list_2[:, 1], points_list_2[:, 2]

    # Creating plot
    # fig = plt.Figure()
    dot_size = 1
    axes_lim = 20

    plt.scatter(x_pts_1, z_pts_1, color="red", s=dot_size)
    plt.scatter(x_pts_2, z_pts_2, color="blue", s=dot_size)

    plt.title("triangulated world points")
    plt.xlim(-axes_lim, axes_lim)
    plt.ylim(-5, 30)
    plt.xlabel("x (dimensionless)")
    plt.ylabel("z (dimensionless)")
    plt.legend(["Linear", "Nonlinear"])

    # show plot
    plt.show()


def visualize_points_camera_poses(points_list, R_set, C_set):

    points_list = np.asarray(points_list)               # use numpy to efficiently get all rows of a column
    # points_list_2 = np.asarray(points_list_2)

    x_pts_1, y_pts_1, z_pts_1 = points_list[:, 0], points_list[:, 1], points_list[:, 2]
    # x_pts_2, y_pts_2, z_pts_2 = points_list_2[:, 0], points_list_2[:, 1], points_list_2[:, 2]

    # Creating plot
    # ax = plt.axes(projection="3d")
    # fig = plt.Figure()
    dot_size = 1
    axes_lim = 20

    plt.scatter(x_pts_1, z_pts_1, color="blue", s=dot_size)
    # plt.scatter(x_pts_2, z_pts_2, color="blue", s=dot_size)

    for i in range(len(R_set)):

        r2 = Rotation.from_matrix(R_set[i])
        angles2 = r2.as_euler("zyx", degrees=True)

        plt.plot(C_set[i][0], C_set[i][2], marker=(3, 0, int(angles2[1])), markersize=15, linestyle='None')

    plt.title("triangulated world points")
    plt.xlim(-axes_lim, axes_lim)
    plt.ylim(-5, 30)
    plt.xlabel("x (dimensionless)")
    plt.ylabel("z (dimensionless)")
    # plt.legend(["Linear", "Nonlinear"])

    # show plot
    plt.show()


def visualize_points_3D(points_list_1, points_list_2, points_list_3, points_list_4):

    points_list_1 = np.asarray(points_list_1)  # use numpy to efficiently get all rows of a column
    points_list_2 = np.asarray(points_list_2)
    points_list_3 = np.asarray(points_list_3)
    points_list_4 = np.asarray(points_list_4)

    x_pts_1, y_pts_1, z_pts_1 = points_list_1[:, 0], points_list_1[:, 1], points_list_1[:, 2]
    x_pts_2, y_pts_2, z_pts_2 = points_list_2[:, 0], points_list_2[:, 1], points_list_2[:, 2]
    x_pts_3, y_pts_3, z_pts_3 = points_list_3[:, 0], points_list_3[:, 1], points_list_3[:, 2]
    x_pts_4, y_pts_4, z_pts_4 = points_list_4[:, 0], points_list_4[:, 1], points_list_4[:, 2]

    # Creating plot
    # fig = plt.Figure()
    ax = plt.axes(projection="3d")
    dot_size = 1
    axes_lim = 30

    ax.scatter3D(x_pts_1, z_pts_1, y_pts_1, color="red", s=dot_size)
    ax.scatter3D(x_pts_2, z_pts_2, y_pts_2, color="blue", s=dot_size)
    ax.scatter3D(x_pts_3, z_pts_3, y_pts_3, color="green", s=dot_size)
    ax.scatter3D(x_pts_4, z_pts_4, y_pts_4, color="purple", s=dot_size)

    plt.title("triangulated world points")
    plt.xlim(-axes_lim, axes_lim)
    plt.ylim(-axes_lim, axes_lim)

    # show plot
    plt.show()



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

