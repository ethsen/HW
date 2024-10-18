import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    Pc_3d = np.hstack((Pc, np.ones((Pc.shape[0], 1))))
    R,t = Procrustes(Pc_3d, Pw[1:4])
    return R,t


def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    worldCoords = np.array([Y[0],Y[1], Y[-1]])
    imgCoords =np.array([X[0],X[1],X[-1]])
    a = np.linalg.norm((worldCoords[1] - worldCoords[2]),ord=1)
    b = np.linalg.norm((worldCoords[0] - worldCoords[2]),ord=1)
    c = np.linalg.norm((worldCoords[0] - worldCoords[1]),ord=1)
    
    j1= (1/np.linalg.norm(imgCoords[0], ord=2)) * imgCoords[0]
    j2= (1/np.linalg.norm(imgCoords[1], ord=2)) * imgCoords[1]
    j3= (1/np.linalg.norm(imgCoords[2], ord=2)) * imgCoords[2]

    alpha = np.arccos(j2 @ j3.T)
    beta = np.arccos(j1 @ j3.T)
    gamma = np.arccos(j1 @ j2.T)
    consts = [a,b,c,alpha,beta,gamma]
    

    coeff = calcCoeffs(consts)
    sArray = quarticFunc(np.roots(coeff),consts)

    
    imgCoords[0] = sArray[0]*j1
    imgCoords[1]= sArray[1]*j2
    imgCoords[2]= sArray[2]*j3

    imgCentroid = np.mean(imgCoords.T, axis=0)
    worldCentroid =np.mean(worldCoords.T, axis=0)
    imgCentered = imgCentered - imgCentroid[:, np.newaxis]
    worldCentered = worldCentered - worldCentroid[:,np.newaxis]

    u,s,v = np.linalg.svd((imgCentered @ worldCentered.T))

    R = u @ v
    t = np.mean(worldCentered.T - R@imgCentered, axis=1)



    return R, t

def calcCoeffs(consts):
    a = consts[0]
    b = consts[1]
    c = consts[2]
    alpha = consts[3]
    beta =consts[4]
    gamma = consts[5]


    a4 = ((a**2 - c**2)/b**2 - 1)**2 - ((4*c**2)/b**2)* np.cos(gamma)**2
    a3 = 4* (((a**2 - c**2)/b**2)(1- (a**2 - c**2)/b**2)*np.cos(beta)- (1-(a**2 + c**2)/b**2)*np.cos(alpha)*np.cos(gamma)
             + 2* (c**2 /b**2)*np.cos(alpha)**2 *np.cos(beta))
    a2 =2*(((a**2 - c**2)/b**2)**2 - 1 +2*(((a**2 - c**2)/b**2)**2)*np.cos(beta)**2
           +2*((b**2 - c**2)/b**2)*np.cos(alpha)**2 - 4*((a**2 + c**2)/b**2)*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
           +2*((b**2 - a**2)/b**2)*np.cos(gamma)**2)
    a1 = 4* ((-(a**2 - c**2)/b**2) * (1 + (a**2 - c**2)/b**2)*np.cos(beta)+ 2*(a**2 /b**2)*np.cos(beta)*np.cos(gamma)**2
             - (1 -((a**2 + c**2)/b**2))*np.cos(alpha)*np.cos(gamma))
    a0 = (1 + (a**2 - c**2)/b**2) - 4*(a**2 /b**2)*np.cos(gamma)**2

    return [a4,a3,a2,a1,a0]

def quarticFunc(roots,consts):
    a = consts[0]
    b = consts[1]
    c = consts[2]
    alpha = consts[3]
    beta =consts[4]
    gamma = consts[5]

    for root in roots:
        u = (((-1 + (a**2 - c**2)/b**2)*root**2 
              - 2*( (a**2 - c**2)/b**2)*np.cos(beta)*root + 1 +
                  (a**2 - c**2)/b**2)) /(2*(np.cos(gamma)- root*np.cos(alpha)))
        s1 = b**2 / (1+ root**2  -2*root*np.cos(beta))
        s2 =u*s1
        s3 = root*s1

        if s1 >0 and s2> 0 and s3>0:
            return [s1,s2,s3]


if __name__ == "__main__":
    pc = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    pw = np.array([[0,2,1],[0,4,1],[4,0,1],[4,4,1]])

    Procrustes(pc,pw)