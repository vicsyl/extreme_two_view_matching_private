# Results and plan

 * Achieved good enough(?) results for the rectification
   * Diff in relative R (T is unscaled!!!)
   * Results 

With rectification

| difficulty | % of correct (with rect)       | % of correct (without rect.)   |
|------------| ------------------------------ | -------------------------------|
|  0         | 0.9                            | 0.915                          |
|  1         | 0.82                           | 0.79
|  2         | 0.7                            | 0.68                           |
|  3         | 0.635                          | 0.43                           |
|  4         | 0.465                          | 0.27                           |
|  5         | 0.42                           | 0.175                          |
|  6         | 0.34                           | 0.12                           |
|  7         | 0.34                           | 0.02                           |
|  8         | 0.215                          | 0.025                          |
|  9         | 0.245                          | 0.015                          |
|  10        | 0.125                          | 0.0                            |
|  11        | 0.145                          | 0.0                            |
|  12        | 0.07                           | 0.0                            |
|  13        | 0.06                           | 0.0                            |
|  14        | 0.01                           | 0.01                           |
|  15        | 0.015                          | 0.005                          |
|  16        | 0.025                          | 0.0                            |
|  14        | 0.021                          | 0.016                          |




# The goals (Q)

 * show the baseline (more or less done) and compare with improvements/other approaches ...
 * baseline + orthogonality + check the original code - (sky segmentation; ?lower level improvements) 
   * matching combinations of pairs of the dominant planes (P1)
     * https://github.com/danini/progressive-x/blob/master/examples/example_multi_homography.ipynb
     * x-progressive mat      
   * planar homology
   * search to match the 3D scenes inferred from the depth maps (A: not simple enough)
     
   * 2-3 for testing and the other for validation 
 * using ...
   * different features (? own implementation?) - (A: pluggable, not now)    
   * different depth maps (A: )
   * different dataset (non orthogonal planes) 
     * (A: The GoogleUrban dataset at https://www.cs.ubc.ca/research/image-matching-challenge/2021/data/ )
     * https://github.com/ubc-vision/image-matching-benchmark/tree/master/utils      
   * (Q) own implementation - RANSAC, features, rectification
 * measuring 
   * error in relative R 
   * '#' keypoints in line with the ground truth (YES!)
   * performance (?) => or relative performance (NOT IMPORTANT)
   

## Opportunities to improve the existing code

### Existing parameters
 * normals computation
   * weighing the SVD (P5)
 * normals clustering (P2)
   * minimal size of the clusters
     maximal distance from the cluster centers
   * enforce the orthogonality(?)
 * rectification (P2)
   * resizing of the patch
   * resizing of the patch 
   * maximal distance from the cluster centers
 * (Q) how to work with the (hyper)params?
   * direct optimization (Bayesian optimization, G. Descent)
   * calculate/estimate somehow
   * guess (5x5 window)


### Existing logic
 * rectification 
   * TODO instructive notebook (P1)  
   * adjusting the patches (removing the higher frequencies) 
   * rounding....
   

### Other approaches
 * explore the types of failures (repeating, too many kpts, normal estimation)

 * RANSAC 
   * constraints - only consistent matches (plane by plane) are considered
     * half way matching the planes and depth (x-progressive)
   * weight - better keypoints (near edges/corners)
   * inliers - consistent with the geometry (place within the patch)
   

 * https://github.com/cavalli1234/AdaLAM
    * sorting of the points
    * inside RANSAC
 * sampling for DEGENSAC: plane and parallax (cooperation to be discussed)

   

 * dataset img matching challenge Saint Peters square / google urban 
 * guessed/unguessed K -> guessed DEGENSAC: to be send -> find F with the guessed

 https://github.com/ducha-aiki/pydegensac/blob/master/examples/simple-example.ipynb

F, mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, th, 0.999, n_iter, enable_degeneracy_check= True)
th = 0.5

(rect., non-rect.)
invite for Wed after 10AM 



#
 matches = bf.knnMatch(descs1, descs2, k=2)
    # For cross-check
    matches2 = bf.match(descs2, descs1)
    good =[]
    if len(matches) < 10:
        return None, [], []
    for m,n in matches:
        if matches2[m.trainIdx].trainIdx != m.queryIdx:
            continue
        if m.distance < 0.85 *n.distance:
            good.append(m)