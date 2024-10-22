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

    p1= 3
    p2= 2
    p3 = 0
    p4 = 1
    homogenousPix = np.hstack((Pc, np.ones((Pc.shape[0], 1))))
    calibCoords= (np.linalg.inv(K) @ homogenousPix.T).T * K[0,0]

    constants = defineConsts(calibCoords, Pw, p1,p2,p3)
    roots = calcRoots(constants)

    for i in range(len(roots)):
        s= quarticFunc(roots[i], constants)
        pc1 = s[0] * constants[6]
        pc2 = s[1] * constants[7]
        pc3 = s[2]*constants[8]
        Pc =np.array([pc1,pc2,pc3])
        R,t = Procrustes(Pc, [Pw[p1],Pw[p2],Pw[p3]])        

        #projected_point = (K @ (R @ Pw[p4].T + t)).T
        projected_point = K @ (R.T @ (Pw[p4].T - t))
        projected_point /= projected_point[2]
        projected_point = np.array(projected_point.flatten())

        if np.allclose(projected_point, homogenousPix[p4],atol=1e-5):
            print("huh")
            return R,t

def defineConsts(imgCoords, worldCoords,p1,p2,p3):
    a = np.linalg.norm((worldCoords[p2] - worldCoords[p3]))
    b = np.linalg.norm((worldCoords[p1] - worldCoords[p3]))
    c = np.linalg.norm((worldCoords[p1] - worldCoords[p2]))
    
    j1= (1/np.linalg.norm(imgCoords[p1])) * imgCoords[p1]
    j2= (1/np.linalg.norm(imgCoords[p2])) * imgCoords[p2]
    j3= (1/np.linalg.norm(imgCoords[p3])) * imgCoords[p3]

    alpha = np.arccos(np.dot(j2,j3))
    beta = np.arccos(np.dot(j1,j3))
    gamma = np.arccos(np.dot(j1,j2))
    return [a,b,c,alpha,beta,gamma,j1,j2,j3]

def calcRoots(consts):
    a = consts[0]
    b = consts[1]
    c = consts[2]
    alpha = consts[3]
    beta =consts[4]
    gamma = consts[5]

    asq_csq_bsq = (a**2 - c**2)/b**2
    asq_csq_bsqPos = (a**2 + c**2)/b**2
    bsq_csq_bsq = (b**2 - c**2)/b**2
    bsq_asq_bsq = (b**2 - a**2)/b**2
    asq_bsq = (a**2)/b**2
    csq_bsq = (c**2)/b**2
    cosAlpha = np.cos(alpha)
    cosBeta = np.cos(beta)
    cosGamma = np.cos(gamma)

    a4 = (asq_csq_bsq -1)**2 - 4*csq_bsq*(cosAlpha**2)

    a3 = 4 *  (asq_csq_bsq * (1- asq_csq_bsq)*cosBeta -
               (1 - asq_csq_bsqPos)*cosAlpha*cosGamma +
               2*csq_bsq*cosBeta*(cosAlpha**2))
    
    a2 = 2 * (asq_csq_bsq**2 - 1 +2*(asq_csq_bsq**2)*cosBeta**2+
              2*bsq_csq_bsq*(cosAlpha**2)-
              4*asq_csq_bsqPos*cosAlpha*cosBeta*cosGamma +
              2* bsq_asq_bsq*(cosGamma**2))
    a1 = 4 * (-asq_csq_bsq*(1+asq_csq_bsq)*cosBeta +
              2*asq_bsq*(cosGamma**2)*cosBeta -
              (1-asq_csq_bsqPos)*cosAlpha*cosGamma)
    a0 = (1+asq_csq_bsq)**2 - 4*asq_bsq*(cosGamma**2)


    roots = np.roots([a4,a3,a2,a1,a0])
    return roots.real[abs(roots.imag) <1e-5]

def quarticFunc(root,consts):
    a = consts[0]
    b = consts[1]
    c = consts[2]
    alpha = consts[3]
    beta =consts[4]
    gamma = consts[5]
    asq_csq_bsq = (a**2 - c**2)/b**2
    cosAlpha = np.cos(alpha)
    cosBeta = np.cos(beta)
    cosGamma = np.cos(gamma)
    
    u = ((asq_csq_bsq -1)*root**2 -
         2*asq_csq_bsq*cosBeta*root+1+
         asq_csq_bsq)/ (2*(cosGamma - root*cosAlpha))
    
    if b**2 / (1+ root**2  -2*root*np.cos(beta)) < 0:
        return [0,0,0]
    else:
        s1 = np.sqrt(b**2 / (1+ root**2  -2*root*cosBeta))
        s2 =u*s1
        s3 = root*s1
    return [s1,s2,s3]

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

    meanPc = np.mean(X, axis= 0)

    meanPw = np.mean(Y,axis= 0)

    a = Y - meanPw
    b = X - meanPc

    u,s,v = np.linalg.svd(a @ b.T, full_matrices= False)

    detCheck = np.eye(3)
    detCheck[-1,-1]= np.linalg.det(v.T @ u.T)

    R = (v.T @ detCheck @ u.T)

    t = meanPw - R@meanPc

    return R,t 

    



if __name__ == "__main__":
    pc = np.array([[304.28,346.36],[449.04,308.92],[363.24,240.72],[232.29,266.60]])
    pw = np.array([[-.07,-.07,0],[0.07,-.07,0],[.07,.07,0],[-.07,.07,0]])
    K = np.array([[823.8,0,304.8],[0,823.8,236.3],[0,0,1]])

    testVec = np.array([1,2,3])
    print(np.linalg.norm(testVec, ord= 1))
    print(np.linalg.norm(testVec))
    #P3P(pc,pw,K)