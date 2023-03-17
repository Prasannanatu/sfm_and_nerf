"""
RBE/CS Spring 2023: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 2

Authors:
Prasanna Natu (pnatu@wpi.edu)
M.S. Robotics Engineering

Peter Dentch (pdentch@wpi.edu)
B.S./M.S. Robotics Engineering
Worcester Polytechnic Institute
"""

import cv2
import numpy as np

from getInliersRANSAC import *
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import * 
from misc import *
from ExtractCameraPose import *
from LinearTriangulation import *
from PnPRANSAC import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *


def main():
    """

    """

    feature_flags, idx = get_feature_flags()
    print('flags: ', feature_flags.shape)

    image_num = 1
    matched_image_num = 2

    # Get the list of matches points between images 1 and 2
    matched_points_1_2, RGB_vals_1_2 = parse_matches_file(image_num, matched_image_num)
    # print(matched_points_1_2)
    # print(RGB_vals_1_2)

    matched_points_1_2 = np.asarray(matched_points_1_2)     # use numpy to efficiently get all rows of a column
    print('num matched points: ', len(matched_points_1_2))

    # u_v_1 = matched_points_1_2[:, 0]
    # u_v_2 = matched_points_1_2[:, 1]
    #
    # F_1_2 = get_fundamental_matrix(u_v_1, u_v_2)
    # print(F_1_2)

    # visualize_matches(image_num, matched_image_num, matched_points_1_2)

    F, best_matched_points_1_2 = get_inliers_RANSAC(matched_points_1_2)
    print("F: ", F)
    print('num best matched points: ', len(best_matched_points_1_2))

    visualize_matches(image_num, matched_image_num, best_matched_points_1_2)

    K = get_K()
    # print("K: ", K)

    E = get_Essential_Matrix(F, K)
    print('E: ', E)

    C_list, R_list = extract_camera_pose(E)
    # print('C: ', C_list)
    # print('R: ', R_list)

    X_points_poses = []

    for i in range(4):                  # triangulate points for all four possible camera poses
        X_points_i = linear_triangulation(K, C_list[i], R_list[i], best_matched_points_1_2)
        X_points_poses.append(X_points_i)

    # print('X points: ', X_points_poses)

    # Disambiguate between the four calculated camera poses using the chierality condition
    C, R, X_points, index = disambiguate_camera_poses(C_list, R_list, X_points_poses)

    # print('C correct: ', C)
    # print('R correct: ', R)
    # print('index: ', index)

    # visualize_points_2D(X_points_poses[0], X_points_poses[1], X_points_poses[2], X_points_poses[3])

    # Refine the triangulated points using a nonlinear estimator with the points as initial conditions
    X_points_corrected = non_linear_triangulation(K, C, R, best_matched_points_1_2, X_points)
    # X_points_corr_unhomo = get_unhomogenous_coordinates(X_points_corrected)

    # print('X points corrected: ', X_points_corrected)
    # visualize_points_lin_nonlin(X_points, X_points_corrected)

    # Perform Perspective-n-Point to estimate the poses of new cameras capturing the scene
    C0 = np.zeros(3)
    R0 = np.eye(3)

    R_set = []
    C_set = []
    X_points_set = []

    R_set.append(R0)
    R_set.append(R)
    C_set.append(C0)
    C_set.append(C)
    X_points_set.append(X_points_corrected)

    image_points_set = []

    # Loop over the remaining images 2 - 5
    for image_num in range(3, 6):

        best_matched_points_1_i = get_inliers(1, image_num)

        u_v_1_12 = best_matched_points_1_2[:, 0]
        u_v_1_1i = best_matched_points_1_i[:, 0]
        u_v_1_i = best_matched_points_1_i[:, 1]

        uv_1i, world_points_1_i = find_matching_points(X_points_corrected, u_v_1_12, u_v_1_1i, u_v_1_i)

        #

        R_new, C_new = PnP_RANSAC(K, uv_1i, world_points_1_i)

        if np.linalg.det(R_new) < 0:    # enforce right-hand coordinate system
            R_new = -R_new
            C_new = -C_new

        R_opt, C_opt = nonlinear_PnP(K, uv_1i, world_points_1_i, R_new, C_new)

        R_set.append(R_opt)
        C_set.append(C_opt)

        X_new_linear = linear_triangulation(K, C_opt, R_opt, best_matched_points_1_i)
        X_points_nonlin = non_linear_triangulation(K, C_opt, R_opt, best_matched_points_1_i, X_new_linear)
        X_points_set.append(X_points_nonlin)

        # Visibility matrix
        # V = build_visibility_matrix(X_points_set[0], feature_flags, image_num, idx)
        # print("V mat for image ", str(image_num), " : ", V)

    visualize_points_camera_poses(X_points_set[0], R_set, C_set)







    




if __name__ == "__main__":
    main()