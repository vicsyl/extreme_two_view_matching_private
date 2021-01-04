import numpy as np
import cv2 as cv
from scene_info import *
import matplotlib.pyplot as plt

if __name__ == "__main__":

    img_pairs = read_image_pairs("scene1")

    img_pair: ImagePairEntry = img_pairs[0][0]

    img1 = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img1))
    img2 = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img2))

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    img_sift_keypoints1 = cv.drawKeypoints(img1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_sift_keypoints2 = cv.drawKeypoints(img2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imwrite('work/sift_keypoints_{}.jpg'.format(img_pair.img1), img_sift_keypoints1)
    cv.imwrite('work/sift_keypoints_{}.jpg'.format(img_pair.img2), img_sift_keypoints2)

    # TODO

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()

    print("done")



