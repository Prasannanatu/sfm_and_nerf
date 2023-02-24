import numpy as np
import matplotlib.pyplot as plt
import random
import math


def disambiguatecamerapose(P,X_final, R_n, T_n,C_n):
    """
    Input : Input would be the the camera pose and solution
    
    Output: The disambiguated camera pose Rotation and final world coordinates

            for the X to be valid  r3 * (X-C) must be valid
    
    """
    total_count = []                    # create the empty list for getting the count of true values for all the possible poses

    for i in range(len(R_n)):           #Looping on 4 possible X, C, R values

        R= R_n[i]                       #Getting the current R value

        r3 = R[:,2]                     #Getting the current r3 value
        
        X = X_final[:,i]                #Getting the current X value
        
        C = C_n[i]                      #Getting the current C value
        
        count = 0                       # initializing a counter
        
        for i in range(X.shape[0]):     # Looping over the entire value for checking  condition 
        
            if (r3 * (X-C)):            # Checking the condition  cheirality
        
                count  = count + 1      # If yes consider it 

        
        total_count.append(count)       # append all values to a list


    
    total_count = np.array(total_count) # converting the list to array.
    
    idx = np.argmax(total_count)        # get the max idx of the all. 

    R_cheiral = R[idx]                  # get the final value of R.
    
    C_cheiral = C[idx]                  # get the final value of Camera Pose between two camera.
    
    X_cheiral = X[:, idx]               # get the final coordinates of the World Points.


    return R_cheiral, C_cheiral, X_cheiral






    

        
        
