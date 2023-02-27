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

    # print('num matched points: ', len(matched_points_1_2))

    # u_v_1 = matched_points_1_2[:, 0]
    # u_v_2 = matched_points_1_2[:, 1]

    # F_1_2 = get_fundamental_matrix(u_v_1, u_v_2)
    # print(F_1_2)

    F = get_inliers_RANSAC(matched_points_1_2)
    print(F)




if __name__ == "__main__":
    main()