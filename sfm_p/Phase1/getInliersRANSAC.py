"""
Computing the best fundamental matrix from the matching features using RANSAC
"""


def parse_matches_file(image_num, matched_image_num):
    """
    Parse matches from the "matching*.txt files provided
    image_num: the '*' in the matching*.txt which is the image number for the provided feature matches, integer 1-4
    matched_image_num: the image number for the matches with image_num to return in a list
    returns the list of matched feature points (u, v) and the list of RGB pixel values.
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

