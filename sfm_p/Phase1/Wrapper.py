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


def main():
    """

    """

    image_num = 2
    matched_image_num = 5

    matched_points_1_2, RGB_vals_1_2 = parse_matches_file(image_num, matched_image_num)

    print(matched_points_1_2)
    print(RGB_vals_1_2)



if __name__ == "__main__":
    main()