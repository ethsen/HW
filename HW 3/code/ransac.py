from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None
    e3 = np.array([[0,-1,0],
                   [1,0,0],
                   [0,0,0]])

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]
        """ YOUR CODE HERE
        """
        E = least_squares_estimation(X1[sample_indices], X2[sample_indices])

        numerator = (X2[test_indices] @ (E @ X1[test_indices].T))**2
        denominator = np.linalg.norm((e3 @ (E @ X1[test_indices].T)), axis= 0)**2
        d1 =  np.diag(numerator /denominator)

        numerator = (X1[test_indices] @ (E @ X2[test_indices].T))**2
        denominator=np.linalg.norm((e3 @ (E.T @ X2[test_indices].T)), axis= 0)**2
        d2 = np.diag(numerator/denominator)
        d = d1 
        #numerator = (np.sum(X2[test_indices]  * (X1[test_indices] @ E), axis=1))**2
        #denominator = np.sum((X1[test_indices] @ E.T)[:,:2]**2, axis= 1) + np.sum((E @ X1[test_indices].T)[:2,:]**2, axis = 0)
        #d = numerator /denominator
        inliers = test_indices[d< eps]
        num_inliers = len(inliers)

        """ END YOUR CODE
        """
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_E = E
            best_inliers = inliers


    return best_E, best_inliers