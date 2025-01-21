import numpy as np


def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
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


if __name__ == "__main__":
    # You could run this file to test out your est_homography implementation
    #   $ python est_homography.py
    # Here is an example to test your code, 
    # but you need to work out the solution H yourself.
    X = np.array([[0, 0],[0, 10], [5, 0], [5, 10]])
    Y = np.array([[3, 4], [4, 11],[8, 5], [9, 12]])
    H = est_homography(X, Y)
    print(H)
    