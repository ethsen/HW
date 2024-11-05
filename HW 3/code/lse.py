import numpy as np

def least_squares_estimation(X1, X2): 
  c1 = X2[:,0][:,np.newaxis] * X1
  c2 = X2[:,1][:,np.newaxis] * X1
  c3 = X2[:,2][:,np.newaxis] * X1
  A = np.hstack([c1,c2,c3])
  u, s, vT = np.linalg.svd(A, full_matrices= True)
  E = vT[-1].reshape(3,3)
  u,s, vT = np.linalg.svd(E)
  #E = u @ np.diag([(s[0] + s[1])/2 , (s[0] + s[1])/2 , 0]) @ vT
  E = u @ np.diag([1 , 1 , 0]) @ vT

  return E
