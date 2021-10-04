# Backlog - plan

## Key idea - isolate the stages

Try the next stage (normal estimation, clustering) with GT input from the previous stage. The performance if the various stages can be tuned independently.
 
  
### Depth estimation  
   * how good does the CNN estimate the depth 
     * try on some dataset with ground truth (MegaDepth DS, COLMAP) - is there any bias in the plane directions normals 
     * try synthetic data of depth maps
   * does to error follow correspond to some probabilistic model
   * rather low priority
    

### Normals estimation & clustering  
   * normals estimation - robustness to local noise 
   * normals clustering - robustness to local noise + capture spatial structure
        - mean-shift
		- bilateral filter (~ generalization of the plane detection)
        - how does enforcing the orthogonality help (to estimate the normals x the matching)  
   * parameters tuning - bayesian optimization / probabilistic model (informed grid search for now)
   * current focus 


### Features rectification & matching 
   * probably most important & potentially creative
   * even if all previous steps are correct, how well does it serve to matching - key concern
   * observe metrics and save (tentatives, #inliers, #inlier ratio)  
   * establish correlation between plane1(t1) <-> plane2(t2) on all tentatives
   * modification of the rectifications
     * idea: rectify only partly
        * rotate only by c.&#945; c &#8712; (0, 1) 
        * rotate to align the camera optical axes - many variants (around 1,2,3 axes)
        * others (e.g. use the estimated dR from matching for the rectification)
   * modifications to the matching 
     * add also the unrectified keypoints on the detected planes
   * ASIFT
   * things already identified - "scaling" by the rectification,  clipping angle
 
