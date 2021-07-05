#import cv2 as cv
import numpy as np

from adalam import AdalamFilter
from pipeline import *
from example import get_matches
from rectification import get_mask_for_components
#import numpy as np
#from matching import find_correspondences, find_correspondences


def try_image_pair(img1_name, components1, normal1, img2_name, components2, normal2, without_mask, area_ratio=100, search_expansion=4):

    print("try_image_pair parameters: {}_{}: {}, {}, {}".format(img1_name, img2_name, without_mask, area_ratio, search_expansion))

    Timer.start()

    pipeline = Pipeline.configure("notebook_configs/adalam_config_whole.txt", None)
    pipeline.start()

    # img1 = "frame_0000001865_1"
    # img2 = "frame_0000001355_1"

    img_pair, _ = pipeline.scene_info.find_img_pair_from_imgs(img1_name, img2_name)

    pipeline.matching_pairs = ["{}_{}".format(img1_name, img2_name)]
    pipeline.run_matching_pipeline()
    #
    imd1 = pipeline.process_image(img1)
    imd2 = pipeline.process_image(img2)

    if normal1 is not None:
        components1 = {key for key, value in imd1.valid_components_dict.items() if value == normal1}

    if normal2 is not None:
        components2 = {key for key, value in imd2.valid_components_dict.items() if value == normal2}

    # label1 = 0
    # label2 = 0
    #
    # components1 = {key for key, value in imd1.valid_components_dict.items() if value == label1}
    # components2 = {key for key, value in imd2.valid_components_dict.items() if value == label2}

    # components1 = {9}
    # components2 = {13}

    h1, w1 = imd1.img.shape[:2]
    h2, w2 = imd2.img.shape[:2]

    mask1 = get_mask_for_components(imd1.components_indices, w1, h1, components1, imd1.pts)
    mask2 = get_mask_for_components(imd2.components_indices, w2, h2, components2, imd2.pts)

    if without_mask:
        mask1 = np.ones(mask1.size, dtype=bool)
        mask2 = np.ones(mask2.size, dtype=bool)

    imd1.rect_pts = imd1.rect_pts[mask1]
    imd1.pts = imd1.pts[mask1]
    imd1.ors = imd1.ors[mask1]
    imd1.scs = imd1.scs[mask1]
    imd1.descriptions = imd1.descriptions[mask1]

    imd2.rect_pts = imd2.rect_pts[mask2]
    imd2.pts = imd2.pts[mask2]
    imd2.ors = imd2.ors[mask2]
    imd2.scs = imd2.scs[mask2]
    imd2.descriptions = imd2.descriptions[mask2]

    k1 = imd1.rect_pts
    p1 = imd1.pts
    o1 = imd1.ors
    s1 = imd1.scs
    d1 = imd1.descriptions

    k2 = imd2.rect_pts
    p2 = imd2.pts
    o2 = imd2.ors
    s2 = imd2.scs
    d2 = imd2.descriptions

    matcher = AdalamFilter()
    # default: 100
    matcher.config['area_ratio'] = area_ratio # 100 #10
    # default 4
    matcher.config['search_expansion'] = search_expansion #4 # 20

    matches = matcher.match_and_filter(k1=k1, k2=k2,
                                       o1=o1, o2=o2,
                                       d1=d1, d2=d2,
                                       s1=s1, s2=s2,
                                       im1shape=imd1.img.shape[:2], im2shape=imd2.img.shape[:2]).cpu().numpy()

    m1 = imd1.pts[matches[:, 0]]
    m2 = imd2.pts[matches[:, 1]]

    vis = get_matches(imd1.img, imd2.img, k1=m1[::4], k2=m2[::4])

    plt.figure(figsize=(9, 9))
    plt.title('{}_{}: {} inliers'.format(img1, img2, matches.shape[0]))

    plt.imshow(vis)
    plt.show()

    thresholds = [1, 0.5, 0.1]
    checks = evaluate_tentatives_against_ground_truth(pipeline.scene_info, img_pair, m1, m2, thresholds=thresholds)
    print("AdaLAM with rectified keypoints - ground truth compatible matches (th: [{}]): {}".format(thresholds, checks))
    print()

    E, inlier_mask, src_pts, dst_pts, tentative_matches = match_epipolar(
        imd1.img,
        imd1.key_points,
        imd1.descriptions,
        imd1.real_K,
        imd2.img,
        imd2.key_points,
        imd2.descriptions,
        imd2.real_K,
        find_fundamental=pipeline.estimate_k,
        img_pair=img_pair,
        out_dir=None,
        show=pipeline.show_matching,
        save=pipeline.save_matching,
        ratio_thresh=pipeline.knn_ratio_threshold,
        ransac_th=pipeline.ransac_th,
        ransac_conf=pipeline.ransac_conf,
        ransac_iters=pipeline.ransac_iters
    )

    evaluate_matching(pipeline.scene_info,
                  E,
                  imd1.key_points,
                  imd2.key_points,
                  tentative_matches,
                  inlier_mask,
                  img_pair,
                  {},
                  imd1.normals,
                  imd2.normals,
                  )

if __name__ == "__main__":

    img_pairs = [ "frame_0000001865_1_frame_0000001355_1",
                  "frame_0000000545_3_frame_0000001555_1",
                  "frame_0000000430_4_frame_0000001525_1",
                  "frame_0000001910_1_frame_0000001345_1",
                  "frame_0000000660_4_frame_0000001095_3",
                  "frame_0000000700_2_frame_0000000700_1",
                  "frame_0000000975_3_frame_0000000720_1",
                  "frame_0000001225_1_frame_0000000650_3",
                  "frame_0000001165_1_frame_0000000390_2",
                  "frame_0000001315_1_frame_0000001910_1"]
    components1 = [None] * len(img_pairs)
    components2 = [None] * len(img_pairs)
    components1[0] = {9}
    components2[0] = {13}

    normals1 = [None] * len(img_pairs)
    normals2 = [None] * len(img_pairs)

    normals1 = [None, 0, 0, 0, 0, 0]
    normals2 = [None, 1, 0, 0, 0, 0]

    for i, img_pair in enumerate(img_pairs[3:4]):

        # img1 = "frame_0000001865_1"
        # img2 = "frame_0000001355_1"
        img1 = img_pair[:18]
        img2 = img_pair[19:]
        print("{} : {}".format(img1, img2))

        try_image_pair(img1, components1[i], normals1[i], img2, components2[i], normals2[1], without_mask=True, area_ratio=100, search_expansion=4)
        try_image_pair(img1, components1[i], normals1[i], img2, components2[i], normals2[1], without_mask=True, area_ratio=10, search_expansion=20)
        # try_image_pair(img1, components1[i], normals1[i], img2, components2[i], normals2[1], without_mask=False, area_ratio=100, search_expansion=4)
        # try_image_pair(img1, components1[i], normals1[i], img2, components2[i], normals2[1], without_mask=False, area_ratio=10, search_expansion=20)
