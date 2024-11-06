import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"
  """ YOUR CODE HERE
  """

  u, s, vT = np.linalg.svd(E, full_matrices= True)
  thetas = [np.pi/2, -np.pi/2]
  uVals = [1,-1]
  for uVal in uVals:
    T = uVal * u[:,2]
    for theta in thetas:
      rot = np.array([[0,-np.sin(theta),0],[np.sin(theta),0,0],[0,0,1]])
      R =  u @ rot.T @ vT
      candidate = {'T': T, 'R':R}
      transform_candidates.append(candidate)

  """ END YOUR CODE
  """
  return transform_candidates