import cv2
import numpy as np

from getInliersRANSAC import *
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import * 
from misc import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
import matplotlib
matplotlib.use('tkagg')
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
import time



def compute_camera_indices(visibility_matrix):
    camera =[]
    points = []

    m,n = visibility_matrix.shape

    for i in range(m):
        for j in range(n):
            if visibility_matrix[i,j] == 1:

                camera.append(j)
                points.append(i)

    camera = np.array(camera).reshape(-1)
    points = np.array(points).reshape(-1)


    return camera, points 



def compute_2D_points(x_index, visibility_matrix, feature_1, feature_2 ):

    points2d = []
    
    v_featutre_1 = feature_1[x_index]
    v_featutre_2 = feature_2[x_index]

    m,n = visibility_matrix.shape

    for i in range(m):
        for j in range(n):
            if visibility_matrix[i,j] == 1:
                point = np.hsatck((v_featutre_1[i,j], v_featutre_2[i,j]))
                points2d.append(point)

    return np.array(points2d).reshape(-1,2)




def get_bundle_adjustment_sparsity(X_f, fitlered_feature_flag, C):
    Cn = Cn + 1
    x_n, visibility_matrix = x_n #Visibility Matrixx
    observations = np.sum(visibility_matrix)
    m = observations * 2
    n = Cn *6 + len(x_n) * 3


    A = lil_matrix((m,n) ,dtype = int)

    i = np.arange(observations)  
    c_i, p_i = compute_camera_indices(visibility_matrix)

    for j in range(6):
        A[2 *i, c_i *6 + j] = 1
        A[2*i + 1, c_i * 6 +j] = 1

    for k in range(3):
        A[2 *i, Cn *6 + p_i * 3 + j] = 1
        A[2*i + 1, Cn * 6 + p_i *3 +j] = 1


    return A

def project3D_2D(p3d, R,C, K):
    I_C = np.hstack((np.identity(3), -C.reshape((3,1))))
    P = np.dot(K, np.dot(R, I_C))
    x = np.hstack((p3d, 1))
    x2d = np.dot(P, x)
    x2d /= x2d[-1]
    return x2d


def Projecting(p3d, K_values, K):
    x2d_ = []

    for i in range(len(K_values)):
        R = get_Rotation(K_values[i,:3], 'e')
        C = K_values[i,3:].reshape(3,1)
        pt3d = p3d[i]
        p2d_ = project3D_2D(pt3d, R,C , K)[:2]
        x2d_.append(p2d_)




def error(x0, Cn, n_points, c_i, p_i, p2d, K):
    Cn = Cn +1 
    K_values = x0[: Cn * 6].reshape((Cn, 6))
    p3d = x0[Cn * 6 :].reshape((n_points,3))
    porjected_points = Projecting(p3d[p_i], K_values[c_i], K)
    error  = (p2d - porjected_points).ravel()

    return error



def bundle_adjustment(X_index , X_all, X_f , feature_1 , feature_2, filtered_flag, R_s, C_s, K, Cn, visibility_matrix, ):
    p3d = X_all[X_index]
    p2d = compute_2D_points(X_index,visibility_matrix,feature_1,feature_2)

    RC = []
    for i in range(Cn+1):
        C, R = C_s[i], R_s[i]
        Q = Rotation_on_euler(R)
        RC_ = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC.append(RC_)
    RC = np.array(RC, dtype=object).reshape(-1,6)

    x0 = np.hstack((RC.ravel(), p3d.ravel()))
    no_of_points = p3d.shape[0]

    camera_indices, points_indices = compute_camera_indices(visibility_matrix)

    A = get_bundle_adjustment_sparsity(X_f,filtered_flag,Cn)
    t0 = time.time()
    res = least_squares(error,x0,jac_sparsity=A, verbose=2,x_scale='jac', ftol=1e-10, method='trf',
                        args=(Cn, no_of_points, camera_indices, points_indices, p2d,K))

    t1 = time.time()
    print("Time required to run Bundle Adj: ", t1-t0, "s \nA matrix shape: ",A.shape,"\n######")

    x1 = res.x
    no_of_cams = Cn + 1
    optim_cam_param = x1[:no_of_cams*6].reshape((no_of_cams,6))
    optim_pts_3d = x1[no_of_cams*6:].reshape((no_of_points,3))

    optim_X_all = np.zeros_like(X_all)
    optim_X_all[X_index] = optim_pts_3d

    optim_C_s , optim_R_s = [], []
    for i in range(len(optim_cam_param)):
        R = get_Rotation(optim_cam_param[i,:3], 'e')
        C = optim_cam_param[i,3:].reshape(3,1)
        optim_C_s.append(C)
        optim_R_s.append(R)

    return optim_X_all, optim_R_s, optim_C_s



