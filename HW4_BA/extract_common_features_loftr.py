import torch
import cv2
import numpy as np
from functools import reduce
import os
import pycolmap
from loftr import LoFTR, default_cfg

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")



def extract_features(matcher, image_pair, filter_with_conf=True):
    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (1280,960))
    img1_raw = cv2.resize(img1_raw, (1280,960))

    img0 = torch.from_numpy(img0_raw)[None][None].to(device, dtype=torch.float32) / 255.0
    img1 = torch.from_numpy(img1_raw)[None][None].to(device, dtype=torch.float32) / 255.0
    batch = {'image0': img0, 'image1': img1}

    #############################  TODO 4.4 BEGIN  ############################
    # Inference with LoFTR and get prediction
    # The model `matcher` takes a dict with keys `image0` and `image1` as input,
    # and writes fine features back to the same dict.
    # You can get the results with keys:
    #   key         :   value
    #   'mkpts0_f'  :   matching feature coordinates in image0 (N x 2)
    #   'mkpts1_f'  :   matching feature coordinates in image1 (N x 2)
    #   'mconf'     :   confidence of each matching feature    (N x 1)
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        if filter_with_conf:
            mconf = batch['mconf'].cpu().numpy()
            mask = mconf > 0.5  # filter feature with confidence higher than 0.5
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]

    #############################  TODO 4.4 END  ############################
        print("matches:", mkpts0.shape[0])
        # RANSAC options
        ransac_options = pycolmap.RANSACOptions()
        ransac_options.max_error = 2.0  # Max pixel reprojection error
        ransac_options.confidence = 0.99  # Confidence level
        ransac_options.min_num_trials = 100  # Minimum RANSAC iterations
        ransac_options.max_num_trials = 1000  # Maximum RANSAC iterations
        inliers = pycolmap.estimate_fundamental_matrix(mkpts0, mkpts1, ransac_options)['inlier_mask']
        print("inliers:", np.count_nonzero(inliers))

        mkpts0 = mkpts0[inliers]
        mkpts1 = mkpts1[inliers]
        

        return mkpts0, mkpts1


#############################  TODO 4.5 BEGIN  ############################

def trackFeatures(matcher, imgs, dir,K):
    
    matches = {}
    imgPair = [
            os.path.join(dir, imgs[0]),  
            os.path.join(dir, imgs[1])
        ]
    refFramePts, secondFramePts = extract_features(matcher, imgPair, filter_with_conf=True)

    for i in range(len(refFramePts)):
        matches[tuple(refFramePts[i])] = [refFramePts[i], secondFramePts[i]]

    for i in range(2,len(imgs)):
        imgPair = [
            os.path.join(dir, imgs[0]),  
            os.path.join(dir, imgs[i])
        ]
        refFramePts, currFramePts = extract_features(matcher,imgPair, filter_with_conf= True)
        newMatches = {}
        for j in range(len(refFramePts)):
            key  =tuple(refFramePts[j])
            if key in matches:
                currTrack =  matches[key].copy()
                currTrack.append(currFramePts[j])
                newMatches[key] =currTrack

        matches = newMatches

    features =np.array(list(matches.values()))

    #scale back up pixels
    features[:,:,0] *= 3.15
    features[:,:,1] *= 3.15
    # Convert to calibrated coordinates
    K_inv = np.linalg.inv(K)
    
    # Convert to homogeneous coordinates
    N, F, _ = features.shape
    homogeneous = np.ones((N, F, 3))
    homogeneous[:, :, :2] = features
    
    # Apply inverse camera matrix to get calibrated coordinates
    calibrated = homogeneous @ K_inv.T
    calibrated = calibrated[:, :, :2] / calibrated[:, :, 2:3]
    
    return calibrated
##############################  TODO 4.5 END  #############################

def visualize_features(images_dir, image_names, features):
    """
    Visualize tracked features across all frames
    features: numpy array of shape (N, F, 2) where N is number of features and F is number of frames
    """
    import matplotlib.pyplot as plt
    
    # Read and resize all images
    images = []
    for img_name in image_names:
        img = cv2.imread(os.path.join(images_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1280, 960))
        images.append(img)
    
    # Create subplot for each image
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    
    # Plot features on each image
    for f in range(len(images)):
        axes[f].imshow(images[f])
        # Plot all feature points for this frame
        frame_pts = features[:, f, :]
        axes[f].plot(frame_pts[:, 0], frame_pts[:, 1], 'ro', markersize=2)
        axes[f].axis('off')
        axes[f].set_title(f'Frame {f}')
    
    plt.tight_layout()
    plt.show()


def main():
    # Dataset
    image_dir = './data/pennlogo/'
    image_names = ['IMG_8657.jpg', 'IMG_8658.jpg', 'IMG_8659.jpg', 'IMG_8660.jpg', 'IMG_8661.jpg']
    K = np.array([[3108.427510480831, 0.0, 2035.7826140150432], 
                  [0.0, 3103.95507309346, 1500.256751469342], 
                  [0.0, 0.0, 1.0]])

    # LoFTR model
    matcher = LoFTR(config=default_cfg)
    # Load pretrained weights
    checkpoint = torch.load("weights/outdoor_ds.ckpt", map_location=device, weights_only=True)
    matcher.load_state_dict(checkpoint["state_dict"])
    matcher = matcher.eval().to(device)

    #############################  TODO 4.5 BEGIN  ############################
    # Find common features
    # You can add any helper functions you need
    # Find matches between consecutive pairs
    features = trackFeatures(matcher, image_names, image_dir,K)

    ##############################  TODO 4.5 END  #############################

    np.savez("loftr_features.npz", data=features, image_names=image_names, intrinsic=K)


if __name__ == '__main__':
    main()
