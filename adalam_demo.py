#import cv2 as cv
import numpy as np

from adalam import AdalamFilter
from pipeline import *
from example import get_matches
from rectification import get_mask_for_components
#import numpy as np
#from matching import find_correspondences, find_correspondences


if __name__ == "__main__":

    Timer.start()

    pipeline = Pipeline.configure("notebook_configs/adalam_config_whole.txt", None)
    pipeline.start()

    img1 = "frame_0000001865_1"
    img2 = "frame_0000001355_1"
    img_pair, _ = pipeline.scene_info.find_img_pair_from_imgs(img1, img2)

    imd1 = pipeline.process_image(img1)
    imd2 = pipeline.process_image(img2)

    label1 = 0
    label2 = 0

    components1 = {key for key, value in imd1.valid_components_dict.items() if value == label1}
    components2 = {key for key, value in imd2.valid_components_dict.items() if value == label2}

    components1 = {9}
    components2 = {13}

    h1, w1 = imd1.img.shape[:2]
    h2, w2 = imd2.img.shape[:2]

    mask1 = get_mask_for_components(imd1.components_indices, w1, h1, components1, imd1.pts)
    mask2 = get_mask_for_components(imd2.components_indices, w2, h2, components2, imd2.pts)

    mask1 = np.ones(mask1.size, dtype=bool)
    mask2 = np.ones(mask2.size, dtype=bool)

    k1 = imd1.rect_pts[mask1]
    p1 = imd1.pts[mask1]
    o1 = imd1.ors[mask1]
    s1 = imd1.scs[mask1]
    d1 = imd1.descriptions[mask1]

    k2 = imd2.rect_pts[mask2]
    p2 = imd2.pts[mask2]
    o2 = imd2.ors[mask2]
    s2 = imd2.scs[mask2]
    d2 = imd2.descriptions[mask2]

    matcher = AdalamFilter()
    # default: 100
    matcher.config['area_ratio'] = 10
    # default 4
    matcher.config['search_expansion'] = 20

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

    checks = evaluate_tentatives_agains_ground_truth(pipeline.scene_info, img_pair, m1, m2, thresholds=[1, 0.5, 0.1])
    print(checks)
    print()