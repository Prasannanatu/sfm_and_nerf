import numpy as np
from getInliersRANSAC import *
from misc import *


def build_visibility_matrix(X_points, feature_flags, camera_num, idx):
    """

    """

    X_points = np.asarray(X_points)
    X_points = get_unhomogenous_coordinates(X_points)

    num_features = feature_flags.shape[0]

    X_all = np.zeros((num_features, 3))
    X_all[idx] = X_points
    X_points[idx] = 1

    temporary = np.zeros(num_features, dtype=int)

    for n in range(camera_num + 1):
        temporary = temporary | feature_flags[:, n]
        # for val in feature_flags[:, n]:
        #     if val == 1:


    X_index = np.where((X_points.reshape(-1)) & temporary)

    visibility_matrix = X_points[X_index].reshape(-1, 1)

    for n_cam in range(camera_num + 1):

        visibility_matrix = np.hstack((visibility_matrix, feature_flags[X_index, n_cam].reshape(-1, 1)))

    _, cols = visibility_matrix.shape

    return X_index, visibility_matrix[:, 1:cols]


def get_feature_flags():
    """
    Parse the image files for matching features to create a boolean matrix for forming the visibility matrix
    :return:
    """
    num_images = 5

    path = '../Data/matching'

    feature_flags = []

    # Loop over the images 1 - 5
    for image in range(0, num_images - 1):

        # open the file, 'r' is read only, 'with' closes it automatically
        with open(path + str(image + 1) + '.txt', 'r') as f:

            for i, line in enumerate(f):  # reading line by line with an interator variable

                feature_flag_row = np.zeros((1, num_images))
                feature_flag_row[0, image - 1] = 1

                # matched_points = []  # hold the current pixel coordinates of the matched image features

                line = line.strip()  # remove the newline character at the end of each line in the file
                line = line.split(" ")  # split up the line string by spaces

                if i == 0:  # on the first line of the file
                    num_features = line[1]  # the second array element should be the number of features
                    continue  # move on to the next line in the file, next loop iteration

                # Parsing the data
                num_matches = int(line[0])  # first argument is the number of matches for the current feature

                # r_val = int(line[1])  # arg 2 is the red pixel color value
                # g_val = int(line[2])  # arg 3 is the green pixel color value
                # b_val = int(line[3])  # arg 4 is the blue pixel color value
                #
                # u = float(line[4])
                # v = float(line[5])

                for j in range(num_matches - 1):  # parse out the given number of feature matches

                    line_index = 6 + (j * 3)  # this is determined by how the file is formatted
                    image = int(line[line_index])  # image number for the matched feature

                    feature_flag_row[0, image - 1] = 1

                    # u_matched = float(line[line_index + 1])
                    # v_matched = float(line[line_index + 2])

                    feature_flags.append(feature_flag_row)

    feature_flags = np.asarray(feature_flags, dtype=int).reshape((-1, num_images))

    # compare each image to every other image
    for i in range(0, 4):  # No of Images = 5
        for j in range(i + 1, 5):

            idx = np.where(feature_flags[:, i] & feature_flags[:, j])
            idx = np.array(idx).reshape(-1)

    return feature_flags, idx


