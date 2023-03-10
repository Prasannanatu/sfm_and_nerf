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
    # visualize_matches(image_num, matched_image_num, best_matched_points_1_2)

    K = get_K()
    # print("K: ", K)

    E = get_Essential_Matrix(F, K)
    print('E: ', E)

    C_list, R_list = extract_camera_pose(E)
    # print('C: ', C_list)
    # print('R: ', R_list)

    X_points_1 = linear_triangulation(K, C_list[0], R_list[0], best_matched_points_1_2)
    X_points_2 = linear_triangulation(K, C_list[1], R_list[1], best_matched_points_1_2)
    X_points_3 = linear_triangulation(K, C_list[2], R_list[2], best_matched_points_1_2)
    X_points_4 = linear_triangulation(K, C_list[3], R_list[3], best_matched_points_1_2)

    visualize_points_2D(X_points_1, X_points_2, X_points_3, X_points_4)




    # R, T, P, C = extract_camera_pose(K, E)
    # print('R: ', R)
    # print('T: ', T)
    # print('P: ', P)
    # print('C: ', C)


    # X = linearTriangulation(R,T,P,K,matched_points_1_2[0],matched_points_1_2[1])
    # print(X)






    




if __name__ == "__main__":
    main()