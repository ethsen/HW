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
        inliers = pycolmap.fundamental_matrix_estimation(mkpts0, mkpts1, ransac_options)['inliers']
        print("inliers:", np.count_nonzero(inliers))

        mkpts0 = mkpts0[inliers]
        mkpts1 = mkpts1[inliers]

        return mkpts0, mkpts1


#############################  TODO 4.5 BEGIN  ############################
# Any helper functions you need for this part
def findMatches(matcher, imgName, imageDir):
    matches = []
    for i in range(len(imgName) - 1):
        image_pair = [
            os.path.join(imageDir, imgName[i]),
            os.path.join(imageDir, imgName[i + 1])
        ]
        mkpts0, mkpts1 = extract_features(matcher, image_pair, filter_with_conf=True)
        matches.append((mkpts0, mkpts1))
    return matches

def trackFeats(matches):
    numImgs = len(matches) + 1
    features = []
    
    pt0, pt1 = matches[0]
    curr = {i: [points0[i], pt1[i]] for i in range(len(pt0))}
    
    for i in range(1, len(matches)):
        points0, points1 = matches[i]
        newTracks = {}
        for id, track in curr.items():
            prevPt = track[-1]
            for j in range(len(points0)):
                if np.allclose(prevPt, points0[j], atol=1.0):
                    newTracks = track + [points1[j]]
                    if len(newTracks) == i + 2:  
                        newTracks[id] = newTracks
                    break
        curr = newTracks
    
    # Convert tracks to numpy array
    commonFeatures = []
    for track in curr.values():
        if len(track) == numImgs:  # Only keep complete tracks
            commonFeatures.append(track)
    
    return np.array(commonFeatures)

##############################  TODO 4.5 END  #############################


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
    matches = findMatches(matcher, image_names, image_dir)
    common_features = trackFeats(matches)

    image_pair = [..., ...]
    extract_features(matcher, image_pair, filter_with_conf=True)
    common_features = ...   # N x F x 2
    ##############################  TODO 4.5 END  #############################

    np.savez("loftr_features.npz", data=common_features, image_names=image_names, intrinsic=K)


if __name__ == '__main__':
    main()