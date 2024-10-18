import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """
    pixels = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    t_wc = (t_wc* np.ones((pixels.shape))).T
    Pw = (R_wc @ (np.linalg.inv(K) @ pixels.T)) + t_wc

    scale = t_wc[2] / (R_wc.T[2] @ (np.linalg.inv(K) @ pixels.T))
    Pw = Pw * scale
    return Pw.T
