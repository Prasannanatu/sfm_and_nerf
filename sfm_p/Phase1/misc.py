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
    # //get get_homogenous_coordinates??
    # print(coordinates.shape)
    coordinates = coordinates.reshape((2,1))
    ones =np.ones((1, coordinates.shape[1]))
    homo = np.vstack([coordinates,ones])
    # a_with_zeros = np.vstack([a, np.zeros((1, 3))])
    # print(homo)

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

        
    
    


    