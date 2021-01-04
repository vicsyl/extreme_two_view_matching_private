import numpy as np
import cv2
from scene_info import *
import matplotlib.pyplot as plt
# from copy import deepcopy
from ransac import *

#TODO from verify
def split_points(tentative_matches, kps1, kps2):
    src_pts = np.float32([ kps1[m[0].queryIdx].pt for m in tentative_matches ]).reshape(-1,2)
    dst_pts = np.float32([ kps2[m[0].trainIdx].pt for m in tentative_matches ]).reshape(-1,2)
    return src_pts, dst_pts


def alternative_method(img1, img2, ratio_thresh=0.8, show=True):

    det = cv2.AKAZE_create()
    kps1, descs1 = det.detectAndCompute(img1, None)
    kps2, descs2 = det.detectAndCompute(img2, None)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    knn_matches = matcher.knnMatch(descs1, descs2, 2)

    tentative_matches_structured = []
    tentative_matches_flat = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            tentative_matches_structured.append([m])
            tentative_matches_flat.append(m)

    if show:
        img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, tentative_matches_structured, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3), plt.show()


    src_pts, dst_pts = split_points(tentative_matches_structured, kps1, kps2)
    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)

    # draw_matches(kps1, kps2, tentative_matches,
    #              H.detach().cpu().numpy(),
    #              H_gt,
    #              inl.cpu().numpy(),
    #              cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB),
    #              cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB))

    draw_matches(kps1, kps2, tentative_matches_flat,
             H,
             H,
             inlier_mask,
             img1,
             img2)
             # cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB),
             # cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB))


def sift_1(img1, img2, img_pair):

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    img_sift_keypoints1 = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_sift_keypoints2 = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('work/sift_keypoints_{}.jpg'.format(img_pair.img1), img_sift_keypoints1)
    cv2.imwrite('work/sift_keypoints_{}.jpg'.format(img_pair.img2), img_sift_keypoints2)

    # TODO

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()



if __name__ == "__main__":

    img_pairs = read_image_pairs("scene1")

    img_pair: ImagePairEntry = img_pairs[0][0]

    img1 = cv2.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img1))
    img2 = cv2.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img2))

    alternative_method(img1, img2, ratio_thresh=0.75, show=False)
    #sift_1(img1, img2, img_pair)
    print("done")
