

 # compute normals
 
---> down-sampled depth map
 * compute z-coord from depth   
 * run (weighted) svd on 5x5 window
 * get the normal as V[:, 2]

params:
  
    svd_smoothing = False
    svd_smoothing_sigma = 1.33
    svd_weighted = True
    svd_weighted_sigma = 0.8

---> downsampled normals

 # filter sky
 
---> downsampled normals

out: 

---> filter_mask

 # cluster normals
 
---> downsampled normals, filter_mask
 
params:

    N_points = 300
    angle_distance_threshold_degrees = 30
    distance_inter_cluster_threshold_factor = 2.5
    points_threshold = 20000

---> clusters_representatives, cluster_indices 

 # connected components

---> clusters_representatives, cluster_indices

 * upsample
 * filter clusters 
 * compute connected components having more pixels than a threshold 

params:

    plane_threshold_degrees = 75
    fraction_threshold = 0.03

---> components_indices

 # rectification

---> img, components_indices, clusters_representatives

 * warp components
 * get kpts, desc
 * kpts.pts <- unwarped locations

params:

    bounding_box[0] * bounding_box[1] > 10**8 (not used?)
    coords.shape[1] * 2.0 == bounding_box_size
 
---> all_kps, all_descs

# matching 

---> all_kps, all_descs

 * find tentatives
 * find essential matrix via RANSAC

params:

    do_flann = True
    ratio_threshold = 0.75

---> E, inlier_mask
