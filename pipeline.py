from scene_info import *
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time

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


def find_homography(tentative_matches, kps1, kps2, img1, img2, show=True):

    src_pts, dst_pts = split_points(tentative_matches, kps1, kps2)
    H, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)

    if show:
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

    #matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matcher = cv.BFMatcher()

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


def correctly_matched_point_for_image_pair(inlier_mask, tentative_matches, kps1, kps2, images_info, img_pair):
    matches = [m for (idx, m) in enumerate(tentative_matches) if inlier_mask[idx ,0] == 1]
    kps1_indices = [m.queryIdx for m in matches]
    kps2_indices = [m.trainIdx for m in matches]

    data_point1_ids, mins1 = get_kps_gt_id(kps1, kps1_indices, images_info[img_pair.img1], diff_threshold=100.0)
    data_point2_ids, mins2 = get_kps_gt_id(kps2, kps2_indices, images_info[img_pair.img2], diff_threshold=100.0)

    data_point_ids_matches = data_point1_ids[data_point1_ids == data_point2_ids]
    unique = np.unique(data_point_ids_matches)
    return unique


def get_kps_gt_id(kps, kps_indices, image_entry: ImageEntry, diff_threshold=2.0):

    kps_matches_points = [list(kps[kps_index].pt) for kps_index in kps_indices]
    kps_matches_np = np.array(kps_matches_points)

    image_data = image_entry.data
    data_ids = image_entry.data_point_idxs

    diff = np.ndarray(image_data.shape)
    mins = np.ndarray(kps_matches_np.shape[0])
    data_point_ids = -2 * np.ones(kps_matches_np.shape[0], dtype=np.int32)
    for p_idx, match_point in enumerate(kps_matches_np):
        diff[:, 0] = image_data[:, 0] - match_point[0]
        diff[:, 1] = image_data[:, 1] - match_point[1]
        diff_norm = np.linalg.norm(diff, axis=1)
        min_index = np.argmin(diff_norm)
        min_diff = diff_norm[min_index]
        mins[p_idx] = min_diff
        if min_diff < diff_threshold:
            data_point_ids[p_idx] = data_ids[min_index]

    return data_point_ids, mins


def find_keypoints(scene_name, image_entry: ImageEntry, descriptor):
    #images_info.image_name

    img_path = 'original_dataset/{}/images/{}.jpg'.format(scene_name, image_entry.image_name)
    img = cv.imread(img_path)
    if img is None:
        return None, None
    kps, descs = descriptor.detectAndCompute(img, None)
    return kps, descs

def find_keypoints_match_with_data(scene_name, image_entry: ImageEntry, descriptor, diff_threshold):

    kps, descs = find_keypoints(scene_name, image_entry, descriptor)
    if kps is None:
        return None
    kps_indices = list(range(len(kps)))

    data_point_ids, mins = get_kps_gt_id(kps, kps_indices, image_entry, diff_threshold=diff_threshold)
    data_point_ids = data_point_ids[data_point_ids != -2]
    return data_point_ids


def keypoints_match_with_data(scene_name, diff_threshold, descriptor=cv.SIFT_create(), limit=None):

    images_info = read_images(scene_name)

    existent_ids = 0
    for idx, image_entry_key in enumerate(images_info):
        image_entry = images_info[image_entry_key]
        data_point_ids = find_keypoints_match_with_data(scene_name, image_entry, descriptor, diff_threshold)
        if data_point_ids is None:
            print("Image: {} doesn't exist!!!".format(image_entry.image_name))
        else:
            all_points = len(image_entry.data)
            print("Image: {}, points matches:{}/{}".format(image_entry.image_name, len(data_point_ids), all_points))
            existent_ids = existent_ids + 1
            if limit is not None and existent_ids == limit:
                break


def match_image_pair(img_pair, images_info, descriptor):

    img1 = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img1))
    img2 = cv.imread('original_dataset/scene1/images/{}.jpg'.format(img_pair.img2))

    tentative_matches, kps1, kps2 = find_correspondences(descriptor, img1, img2, ratio_thresh=0.75, show=False)
    H, inlier_mask = find_homography(tentative_matches, kps1, kps2, img1, img2, show=False)

    unique = correctly_matched_point_for_image_pair(inlier_mask, tentative_matches, kps1, kps2, images_info, img_pair)

    print("Image pair: {}x{}:".format(img_pair.img1, img_pair.img2))
    print("Number of correspondences: {}".format(inlier_mask[inlier_mask == [0]].shape[0]))
    print("correctly_matched_point_for_image_pair: unique = {}".format(unique.shape[0]))

    # TODO CONTINUE2
    # verify pose - this seems to be difficult, up next...
    # a = images_info[img_pair.img1].qs
    # b = images_info[img_pair.img1].t
    # c = H


def img_correspondences(scene_name, descriptor=cv.SIFT_create(), difficulties = set(range(18)), limit=None):

    #akaze_desc = cv.AKAZE_create()
    img_pairs = read_image_pairs(scene_name)
    images_info = read_images(scene_name)

    for difficulty, img_pair_in_difficulty in enumerate(img_pairs):
        if difficulty not in difficulties:
            continue
        print("Difficulty: {}".format(difficulty))
        if limit is None:
            limit = len(img_pair_in_difficulty)
        for i in range(min(limit, len(img_pair_in_difficulty))):
            img_pair: ImagePairEntry = img_pairs[difficulty][i]
            match_image_pair(img_pair, images_info, descriptor)


def main():

    start = time.time()

    scene_name = "scene1"
    #akaze_desc = cv.AKAZE_create()
    sift_descriptor = cv.SIFT_create()

    limit = 10
    keypoints_match_with_data(scene_name, 2, sift_descriptor, limit)


    #limit = 3
    #difficulties = set(range(1))
    #img_correspondences(scene_name, sift_descriptor, difficulties, limit)

    print("All done")
    end = time.time()
    print("Time elapsed: {}".format(end - start))


if __name__ == "__main__":
    main()
