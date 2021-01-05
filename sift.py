from scene_info import *
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

"""
    DISCLAIMER: some of the these methods were taken from templates (.py files or jupyter notebooks) 
    for the tasks for the MPV course taken by me (Vaclav Vavra) in the spring semester 2020.
"""


def decolorize(img):
    return cv.cvtColor(cv.cvtColor(img,cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)


def draw_matches(kps1, kps2, tentative_matches, H, inlier_mask, img1, img2):
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, H)
    img2_tr = cv.polylines(decolorize(img2), [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
    matches_mask = inlier_mask.ravel().tolist()

    # Blue is estimated homography
    draw_params = dict(matchColor = (255, 255, 0), # draw matches in yellow color
                   singlePointColor = None,
                   matchesMask = matches_mask, # draw only inliers
                   flags = 20)
    img_out = cv.drawMatches(decolorize(img1), kps1, img2_tr, kps2, tentative_matches, None, **draw_params)
    plt.figure(figsize=(20, 10))
    plt.title("Matches with estimated homography")
    plt.imshow(img_out)
    plt.show()
    return


def split_points(tentative_matches, kps1, kps2):
    src_pts = np.float32([ kps1[m[0].queryIdx].pt for m in tentative_matches ]).reshape(-1,2)
    dst_pts = np.float32([ kps2[m[0].trainIdx].pt for m in tentative_matches ]).reshape(-1,2)
    return src_pts, dst_pts

# TODO split this into ransac and matching
def match_by_ransac(descriptor, img1, img2, ratio_thresh=0.8, show=True):

    kps1, descs1 = descriptor.detectAndCompute(img1, None)
    kps2, descs2 = descriptor.detectAndCompute(img2, None)

    # img_sift_keypoints1 = cv.drawKeypoints(img1, kps1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img_sift_keypoints2 = cv.drawKeypoints(img2, kps2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # cv.imwrite('work/sift_keypoints_{}.jpg'.format(img_pair.img1), img_sift_keypoints1)
    # cv.imwrite('work/sift_keypoints_{}.jpg'.format(img_pair.img2), img_sift_keypoints2)

    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    knn_matches = matcher.knnMatch(descs1, descs2, 2)

    tentative_matches_structured = []
    tentative_matches_flat = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            tentative_matches_structured.append([m])
            tentative_matches_flat.append(m)

    if show:
        img3 = cv.drawMatchesKnn(img1, kps1, img2, kps2, tentative_matches_structured, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3)
        plt.show()


    src_pts, dst_pts = split_points(tentative_matches_structured, kps1, kps2)
    H, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)

    draw_matches(kps1, kps2, tentative_matches_flat, H, inlier_mask, img1, img2)


if __name__ == "__main__":

    akaze_desc = cv.AKAZE_create()
    sift = cv.SIFT_create()

    img_pairs = read_image_pairs("scene1")

    img_pair: ImagePairEntry = img_pairs[0][0]

    img1 = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img1))
    img2 = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img2))

    match_by_ransac(akaze_desc, img1, img2, ratio_thresh=0.75, show=True)
    print("done")
