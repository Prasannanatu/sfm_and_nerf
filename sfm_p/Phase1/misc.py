import numpy as np
import math
import random


def skew_matrix(x):
    X = np.array([[0, -x[2] , x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
    return X


def get_homogenous_coordinates(coordinates):
    """
    Input: The co-ordinates u,v
    Outputs : The homogenize coordinates
    """

    coordinates = np.asarray(coordinates)
    ones = np.ones((coordinates.shape[0], 1))

    homo = np.concatenate((coordinates, ones), axis=1)

    return homo


def get_unhomogenous_coordinates(coordinates):
    """
    
    """

    unhomo = np.delete(coordinates, coordinates.shape[1]-1, axis =1)

    return unhomo


def get_K():
    path = '../Data/calibration.txt'
    K = np.empty((3,3))
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line_numbers = line.strip().split()  # extract the float numbers from the line
            for j, num in enumerate(line_numbers):
                line_number = float(num)
                K[i][j] = line_number
            # line_numbers = [float(num) for num in line_numbers]  # convert the numbers to float
            # K[]

    # k =np.asarray(K)
    # k.reshape((3,3))
    return K

        
    
    


    