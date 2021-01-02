import numpy as np
import cv2 as cv


if __name__ == "__main__":

    img = cv.imread('original_dataset/scene1/images/frame_0000000010_3.jpg')
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    img = cv.drawKeypoints(gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('work/sift_keypoints.jpg', img)
    print("done")



