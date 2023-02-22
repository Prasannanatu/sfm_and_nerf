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
    ones =np.ones(coordinates.shape[1])
    homo = np.concatenate(coordinates,ones, axis = 1)

    return homo


def get_unhomogenous_coordinates(coordinates):
    """
    
    """

    unhomo = np.delete(coordinates, coordiantes.shape[1]-1, axis =1)

    return unhomo
    


    