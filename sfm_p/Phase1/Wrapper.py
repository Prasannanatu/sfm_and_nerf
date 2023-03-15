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
from DisambiguateCameraPose import *
from NonlinearTriangulation import *


def main():
    """

    """

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

    # print('X points corrected: ', X_points_corrected)
    visualize_points_lin_nonlin(X_points, X_points_corrected)

    # Perform Perspective-n-Point to estimate the poses of new cameras capturing the scene






    




if __name__ == "__main__":
    main()