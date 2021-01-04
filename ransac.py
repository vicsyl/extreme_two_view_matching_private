import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


#SECTION
def decolorize(img):
    return  cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

def draw_matches(kps1, kps2, tentative_matches, H,  H_gt, inlier_mask, img1, img2):
    matchesMask = inlier_mask.ravel().tolist()
    h,w, ch = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)
    #Ground truth transformation
    dst_GT = cv2.perspectiveTransform(pts, H_gt)
    img2_tr = cv2.polylines(decolorize(img2),[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
    img2_tr = cv2.polylines(deepcopy(img2_tr),[np.int32(dst_GT)],True,(0,255,0),3, cv2.LINE_AA)
    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor = (255,255,0), # draw matches in yellow color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 20)
    img_out = cv2.drawMatches(decolorize(img1),kps1,img2_tr,kps2,tentative_matches,None,**draw_params)
    plt.figure(figsize=(20, 10))
    plt.imshow(img_out)
    plt.show()
    return




# def verify(tentative_matches, kps1, kps2):
#     src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentative_matches ]).reshape(-1,2)
#     dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentative_matches ]).reshape(-1,2)
#     H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,1.0)
#     return H, inlier_mask
#

#SECTION global code

# H_gt = np.loadtxt('v_woman_H_1_6')
# #Geometric verification (RANSAC)
# H, inliers =  verify(tentative_matches, kps1, kps2)
#
# draw_matches(kps1, kps2, tentative_matches, H, H_gt, inliers, cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB),
#               cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB))



# SECTION

#
# timg1 = timg_load('v_woman1.ppm', True)/255.
# timg2 = timg_load('v_woman6.ppm', True)/255.

# def putting_all_together(img1, img2):
#
#     with torch.no_grad():
#         # keypoint_locations1, descs1, A1 = detect_and_describe(timg1)
#         # keypoint_locations2, descs2, A2 = detect_and_describe(timg2)
#         keypoint_locations1 = None
#         keypoint_locations2 = None
#         desc1 = desc2 = None
#         # kps1 = keypoint_locations_to_opencv_kps(keypoint_locations1)
#         # kps2 = keypoint_locations_to_opencv_kps(keypoint_locations2)
#         kps1 = kps2 = None
#
#         # match_idxs, vals = match_smnn(descs1, descs2, 0.85)
#         # tentative_matches = tentatives_to_opencv(match_idxs, vals)
#         # pts_matches = torch.cat([A1[match_idxs[:,0],:2,2], A2[match_idxs[:,1],:2,2]], dim=1)
#         # H, inl = ransac_h(pts_matches, 4.0, 0.99, 10000)
#         H = inl = None
#         tentative_matches = None
#         H_gt = H
#
#     draw_matches(kps1, kps2, tentative_matches,
#                  H.detach().cpu().numpy(),
#                  H_gt,
#                  inl.cpu().numpy(),
#                  cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB),
#                  cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB))