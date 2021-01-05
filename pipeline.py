from scene_info import *
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

"""
    DISCLAIMER: this function was taken from templates (.py files or jupyter notebooks) 
    for the tasks for the MPV course taken by me (Vaclav Vavra) in the spring semester 2020.
"""
def decolorize(img):
    return cv.cvtColor(cv.cvtColor(img,cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)


"""
    DISCLAIMER: this function was taken from templates (.py files or jupyter notebooks) 
    for the tasks for the MPV course taken by me (Vaclav Vavra) in the spring semester 2020.
"""
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
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentative_matches ]).reshape(-1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentative_matches ]).reshape(-1,2)
    return src_pts, dst_pts


def find_homography(tentative_matches, kps1, kps2):

    src_pts, dst_pts = split_points(tentative_matches, kps1, kps2)
    H, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)

    draw_matches(kps1, kps2, tentative_matches, H, inlier_mask, img1, img2)
    return H, inlier_mask


def find_correspondences(descriptor, img1, img2, ratio_thresh=0.8, show=True):

    kps1, descs1 = descriptor.detectAndCompute(img1, None)
    kps2, descs2 = descriptor.detectAndCompute(img2, None)

    # img_sift_keypoints1 = cv.drawKeypoints(img1, kps1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img_sift_keypoints2 = cv.drawKeypoints(img2, kps2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # cv.imwrite('work/sift_keypoints_{}.jpg'.format(img_pair.img1), img_sift_keypoints1)
    # cv.imwrite('work/sift_keypoints_{}.jpg'.format(img_pair.img2), img_sift_keypoints2)

    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    knn_matches = matcher.knnMatch(descs1, descs2, 2)

    tentative_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            tentative_matches.append(m)

    if show:
        tentative_matches_in_singleton_list = [[m] for m in tentative_matches]
        img3 = cv.drawMatchesKnn(img1, kps1, img2, kps2, tentative_matches_in_singleton_list, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3)
        plt.show()

    return tentative_matches, kps1, kps2


if __name__ == "__main__":

    scene_name = "scene1"

    akaze_desc = cv.AKAZE_create()
    sift = cv.SIFT_create()

    img_pairs = read_image_pairs(scene_name)
    images_info = read_images(scene_name)

    img_pair: ImagePairEntry = img_pairs[0][0]

    img1 = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img1))
    img2 = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img2))

    tentative_matches, kps1, kps2 = find_correspondences(akaze_desc, img1, img2, ratio_thresh=0.75, show=True)
    H, inlier_mask = find_homography(tentative_matches, kps1, kps2)

    # TODO CONTINUE2
    # verify pose - this seems to be difficult, up next...
    a = images_info[img_pair.img1].qs
    b = images_info[img_pair.img1].t
    c = H

    # verify matches
    # inlier_mask_flat = inlier_mask[:, 0]
    # inlier_mask_flat_bool = inlier_mask_flat[inlier_mask_flat == 1]
    matches = [m for (idx, m) in enumerate(tentative_matches) if inlier_mask[idx ,0] == 1]
    kps1_indices = [m.queryIdx for m in matches]
    kps2_indices = [m.trainIdx for m in matches]

    kps1_matches_points = [list(kps1[kps1_index].pt) for kps1_index in kps1_indices]
    kps2_matches_points = [list(kps2[kps2_index].pt) for kps2_index in kps2_indices]

    kps1_matches_np = np.array(kps1_matches_points)
    kps2_matches_np = np.array(kps2_matches_points)

    # TODO CONTINUE1
    # find from data a points closest to coordinates1,2
    data1 = images_info[img_pair.img1].data

    print("done")
