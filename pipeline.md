


 # compute normals
 
---> down-sampled img -> depth map
 * compute z-coord from depth   
 * run (weighted) svd on 5x5 window
 * get the normal as V[:, 2]

 * assumptions: smoothing -> clustering 

params:
  
    5x5 window (~img size)
    svd_weighted = True
    svd_weighted_sigma = 0.8 -> important detail - let's see (maybe grid search)
    (gaussian smoothing = False)
    (gaussian smoothing sigma = 1.33)
    
 * (median/gaussian filter)

---> normals

 # filter sky
 
---> normals

out: 

---> filter_mask

 # cluster normals
 
---> normals, filter_mask
 
* assumptions: smoothing -> clustering/voting

params:

    N_points = 300
    angle_distance_threshold_degrees = 30
    distance_inter_cluster_threshold_factor = 2.5 ("soft enforcement")
        - (60 degrees might be better)
        - weighted based on the distance !!!
        - opposite directions!!!  
    points_threshold = 20000 (~ %3 img size)

 * bilateral filtering - https://en.wikipedia.org/wiki/Bilateral_filter 


---> clusters_representatives, cluster_indices 

 # connected components

---> clusters_representatives, cluster_indices

 * upsample
 * filter clusters 
 * compute connected components having more pixels than a threshold 

* assumptions: smoothing -> clustering

params:

    plane_threshold_degrees = 75
      * try to rectify to 75 degrees max (partly? Fig.2)
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


