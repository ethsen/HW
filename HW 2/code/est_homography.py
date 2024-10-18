import numpy as np

def est_homography(X, Y):
    """ 
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out 
    what X and Y should be. 
    Input:
        X: 4x2 matrix of (x,y) coordinates 
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X
        
    """
    
    A= []
    for i in range(0,4):
        ax = [-X[i,0], -X[i,1],-1,0,0,0,X[i,0]*Y[i,0], X[i,1]*Y[i,0], Y[i,0]]
        ay = [0, 0, 0, -X[i,0], -X[i,1], -1, X[i,0]*Y[i,1], X[i,1]*Y[i,1], Y[i,1]]
        A += [ax] +[ay]
    A = np.array(A)
    u,s,v =  np.linalg.svd(A)
    H = v[-1].reshape(3,3)
    return H/H[-1,-1]

