"""
Computing the best fundamental matrix from the matching features using RANSAC
"""

import numpy as np
from EstimateFundamentalMatrix import *


def parse_matches_file(image_num, matched_image_num):
    """
    Parse matches from the "matching*.txt files provided
    :param image_num: the '*' in the matching*.txt which is the image number for the provided feature matches
    :param matched_image_num: the image number for the matches with image_num to return in a list
    :return: the list of matched feature points (u, v) and the list of RGB pixel values.
    """

    path = '../Data/matching'

    num_features = -1        # initialized to a value it should never be

    RGB_vals = []                       # hold the RGB pixel values of the listed features
    image_feature_points = []           # hold the pixel coordinates of the matched image features

    # open the file, 'r' is read only, 'with' closes it automatically
    with open(path + str(image_num) + '.txt', 'r') as f:

        for i, line in enumerate(f):            # reading line by line with an interator variable

            matched_points = []                 # hold the current pixel coordinates of the matched image features

            line = line.strip()                 # remove the newline character at the end of each line in the file
            line = line.split(" ")              # split up the line string by spaces

            if i == 0:                          # on the first line of the file
                num_features = line[1]          # the second array element should be the number of features
                continue                        # move on to the next line in the file, next loop iteration

            # Parsing the data
            num_matches = int(line[0])          # first argument is the number of matches for the current feature

            r_val = int(line[1])                # arg 2 is the red pixel color value
            g_val = int(line[2])                # arg 3 is the green pixel color value
            b_val = int(line[3])                # arg 4 is the blue pixel color value

            u = float(line[4])
            v = float(line[5])

            for j in range(num_matches - 1):                    # parse out the given number of feature matches

                line_index = 6 + (j * 3)                        # this is determined by how the file is formatted
                image_num = int(line[line_index])               # image number for the matched feature
                u_matched = float(line[line_index + 1])
                v_matched = float(line[line_index + 2])

                if image_num == matched_image_num:              # only append data if from the desired matched image

                    RGB_vals.append([r_val, g_val, b_val])

                    matched_points.append([u, v])
                    matched_points.append([u_matched, v_matched])

                    image_feature_points.append(matched_points)

    return image_feature_points, RGB_vals


def get_inliers(image_1, image_2):

    matched_points, _ = parse_matches_file(image_1, image_2)
    _, refined_points = get_inliers_RANSAC(matched_points)

    return np.asarray(refined_points)



def get_inliers_RANSAC(matched_points):
    """
    Perform the RANSAC algorithm using the fundamental matrix to estimate inlier correspondences between image pairs
    :param matched_points: a list of the matched pixel coordinates of two images, of the form [n x 2 x 2]
    :return: the fundamental matrix with the maximum number of matched point inliers and the new list of best matches
    """

    iterations = 1000                                   # iterations of RANSAC to attempt unless found enough good paris
    epsilon = 0.05                                      # threshold for fundamental matrix transforming
    percent_good_matches = 0.99                         # what percentage of num_matches are enough to stop iterating

    matched_points = np.asarray(matched_points)         # use numpy for efficiently getting all rows of a column
    num_matches = len(matched_points)                   # number of matching feature coordinates between the images

    latest_F = np.zeros((3, 3))                         # latest computed fundamental matrix from the matched pairs

    maximum = 0                                         # how many good matches were found in the last iteration

    best_matches = []                                   # array list of best matches

    for index in range(iterations):                     # index iterator variable non-use intentional

        # pairs_indices = []

        # Select 8 matched feature pairs from each image at random
        points = [np.random.randint(0, num_matches) for num in range(8)]        # 8 random points within num_matches

        pt_1 = matched_points[points[0], 0]
        pt_2 = matched_points[points[1], 0]
        pt_3 = matched_points[points[2], 0]
        pt_4 = matched_points[points[3], 0]
        pt_5 = matched_points[points[4], 0]
        pt_6 = matched_points[points[5], 0]
        pt_7 = matched_points[points[6], 0]
        pt_8 = matched_points[points[7], 0]
        pt_p_1 = matched_points[points[0], 1]               # pt_p is an abbreviation for point_prime
        pt_p_2 = matched_points[points[1], 1]
        pt_p_3 = matched_points[points[2], 1]
        pt_p_4 = matched_points[points[3], 1]
        pt_p_5 = matched_points[points[4], 1]
        pt_p_6 = matched_points[points[5], 1]
        pt_p_7 = matched_points[points[6], 1]
        pt_p_8 = matched_points[points[7], 1]

        pts = np.array([pt_1, pt_2, pt_3, pt_4, pt_5, pt_6, pt_7, pt_8], np.float32)
        pts_prime = np.array([pt_p_1, pt_p_2, pt_p_3, pt_p_4, pt_p_5, pt_p_6, pt_p_7, pt_p_8], np.float32)

        F = get_fundamental_matrix(pts, pts_prime)          # estimate the F matrix using the 8 matching paris

        num_good_matches = 0

        good_matches = []

        # Compute inliers or best matches using |x2 * F * x1| < threshold, repeat until sufficient matches found
        for i in range(num_matches):

            x_pt = matched_points[i, 0, 0]
            y_pt = matched_points[i, 0, 1]
            x_pt_prime = matched_points[i, 1, 0]
            y_pt_prime = matched_points[i, 1, 1]

            # [x, y] to [x, y, 1] for homogeneous coordinates
            point = np.array([x_pt, y_pt, 1], np.float32)
            point_prime = np.array([x_pt_prime, y_pt_prime, 1], np.float32)

            F_pt = F @ point.T
            F_pt = F_pt.T

            pt_prime_F_pt = np.multiply(point_prime, F_pt)

            error = np.sum(pt_prime_F_pt)

            if abs(error) < epsilon:
                # print(pt_prime_F_pt)
                # print('error: ', error)
                # paris_indices.append(i)
                num_good_matches += 1
                good_match = [[x_pt, y_pt], [x_pt_prime, y_pt_prime]]
                good_matches.append(good_match)

        if maximum < num_good_matches:

            maximum = num_good_matches
            print('good matches found: ', num_good_matches)

            # [matches_p.append(matched_points[ind, 0]) for ind in pairs_indices]
            # [matches_p_prime.append(matched_points[ind, 1]) for ind in pairs_indices]

            # matches_p = np.asarray(matches_p)
            # matches_p_prime = np.asarray(matches_p_prime)

            good_matches = np.asarray(good_matches)
            matches_p = good_matches[:, 0]
            matches_p_prime = good_matches[:, 1]

            # Compute the fundamental matrix for the matched pairs
            latest_F = get_fundamental_matrix(matches_p, matches_p_prime)

            # Set for the output array of best matched points
            best_matches = good_matches

            if num_good_matches > percent_good_matches * num_matches:       # end if desired matches num were found
                break

    # best_matches = [matched_points[x] for x in pairs_indices]  # find the pairs corresponding to the computed indicies

    return latest_F, best_matches


def visualize_matches(num_image_1, num_image_2, matched_points):
    """
    Mark out and draw each matched corner pair on the two images
    :param num_image_1: first image number
    :param num_image_2: second image number
    :param matched_points: The array of matched corner points of shape [n x 2 x 2]
    """

    im_path = '../Data/'

    image1 = cv2.imread(im_path + str(num_image_1) + '.png')  # load from image file
    image2 = cv2.imread(im_path + str(num_image_2) + '.png')

    # Handling the case of a color or greyscale image passed to the visualizer
    if len(image1.shape) == 3:
        height1, width1, depth1 = image1.shape
        height2, width2, depth2 = image2.shape
        shape = (max(height1, height2), width1 + width2, depth1)

    elif len(image1.shape) == 2:
        height1, width1, depth1 = image1.shape
        height2, width2, depth2 = image2.shape
        shape = (max(height1, height2), width1 + width2)

    image_combined = np.zeros(shape, type(image1.flat[0]))  # blank side-by-side images
    image_combined[0:height1, 0:width1] = image1  # fill in left side image
    image_combined[0:height1, width1:width1 + width2] = image2  # fill in right side image
    image_1_2 = image_combined.copy()

    circle_size = 4
    red = [0, 0, 255]
    cyan = [255, 255, 0]
    yellow = [0, 255, 255]

    # Mark each matched corner pair with a circle and draw a line between them
    for i in range(len(matched_points)):
        corner1_x = matched_points[i][0][0]
        corner1_y = matched_points[i][0][1]
        corner2_x = matched_points[i][1][0]
        corner2_y = matched_points[i][1][1]

        cv2.line(image_1_2, (int(corner1_x), int(corner1_y)), (int(corner2_x + image1.shape[1]), int(corner2_y)), red,
                 1)
        cv2.circle(image_1_2, (int(corner1_x), int(corner1_y)), circle_size, cyan, 1)
        cv2.circle(image_1_2, (int(corner2_x) + image1.shape[1], int(corner2_y)), circle_size, yellow, 1)

    # Resize for better displaying
    scale = 1.5
    height = image_1_2.shape[0] / scale
    width = image_1_2.shape[1] / scale
    im = cv2.resize(image_1_2, (int(width), int(height)))

    # Save as "".png
    cv2.imwrite('Filtered matches' + '.png', im)
    cv2.imshow("Matches visualizing", im)
    cv2.waitKey(0)

