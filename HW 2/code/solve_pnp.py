from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """
    #Pw = Pw[:,:2]/ Pw[:,-1][:, np.newaxis]
    Pc = np.hstack((Pc, np.ones((Pc.shape[0], 1))))
    Pc = (np.linalg.inv(K) @ Pc.T).T
    #Pw = Pw[:,:2]

    homography =  est_homography(Pw[:,:2], Pc[:,:2])
    hPrime = np.linalg.inv(K) @ homography

    u,s,v = np.linalg.svd(hPrime[:,:2], full_matrices= False)
    r1r2 = u @ v
    r3 = np.cross(r1r2[:,0], r1r2[:,1])

    R = np.array([r1r2[:,0],r1r2[:,1], r3]).T
    lmda = np.sum(s)/2

    t = hPrime[:,-1] /lmda
    
    return R, t


if __name__ == "__main__":
    pc = np.array([[0,0],[0,1],[1,0],[1,1]])

    pw = np.array([[0,2,1],[0,4,1],[4,0,1],[4,4,1]])
    h = np.array([[0,2,1],[0,4,1],[4,0,1]])
    rot,t = PnP(pc,pw)
    
    print(pw[0])
