# Results and plan

 * Achieved good enough(?) results for the rectification
   * Diff in relative R (T is unscaled!!!)
   * Results 

| difficulty | % of correctly estimated poses |
|------------| ------------------------------ |
|  0         | 0.9                            |
|  1         | 0.82                           |
|  2         | 0.7                            |
|  3         | 0.635                          |
|  4         | 0.465                          |
|  5         | 0.42                           |

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
   

 

