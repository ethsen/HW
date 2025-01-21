import numpy as np
from est_homography import est_homography


def warp_pts(X, Y, interior_pts):
    """
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts.
        These coordinate describe where a point inside the goal will be warped
        to inside the penn logo. For this assignment, you can keep these new
        coordinates as float numbers.

    """

    H = est_homography(X, Y)
    interior_pts = np.hstack([interior_pts, np.ones((interior_pts.shape[0],1))])
    warped_pts = np.matmul(H,interior_pts.T).T
    warped_pts = warped_pts[:,:2]/warped_pts[:,2][:,np.newaxis]
    return warped_pts


if __name__ == "__main__":
    # You could run this file to test out your est_homography implementation
    #   $ python est_homography.py
    # Here is an example to test your code, 
    # but you need to work out the solution H yourself.
    X = np.array([[0, 0],[0, 10], [5, 0], [5, 10]])
    Y = np.array([[3, 4], [4, 11],[8, 5], [9, 12]])
    Z = np.array([[1,2], [1,1],[2, 5], [3, 12]])
    print(warp_pts(X,Y,Z))