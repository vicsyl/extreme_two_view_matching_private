# Backlog - plan

## Key idea - isolate the steps

  
 * depth estimation (P>>1) 
   * how good does MD estimate the depth - try on some dataset with ground truth (!)
   * is there any bias in the plane directions?  
   * vs. try synthetic data of depth maps!!!
   * it would be great to establish some model for the errors/stats for the train dataset and use it for e.g. downstream params  

 * normals estimation, clustering (current focus) 
   * normals estimation - robustness to local noise, 
     * parameters tuning - bayesian optimization / probabilistic model
   * normals clustering - robustness to local noise + capture spatial structure
        - mean-shift
        - enforce orthogonality (it "should" help)  
		- bilateral filter (~ generalization of the plane detection)

 * rectification -> features -> matching (I think this should be the most important part)
   * even if all previous steps are correct, how well does it serve to matching - key concern     
   * try to not filter the unrectified keypoints on the detected planes
     * observe metrics (#inliers, #inlier ratio)  
     * establish relation between plane1(t1) <-> plane2(t2) on all tentatives
   * modification of the rectifications
     * idea: rectify only partly
        * only rotation^C (C < 1)
        * rotate to align the camera optical axes - many variants (around 1,2,3 axes)
        * combinations (or e.g. involve the estimated dR and loop back)
   * don't forget about the peculiarities of the rectification
     * "scaling"
     * clipping angle
   * ASIFT
    
 
## Questions:

 * how to improve the process?
   * I am tempted to rewrite it all together
     * clean up the code
     * GPU
     * definitely I think I can rewrite at least some parts
   * have a process for quick experiments and their acceptance/rejection   
    

# Results / done:

 * multiple params for the clustering:
   * observing how close the angle between 2 biggest planes are to the right angle - see the spreadsheets
   * ongoing research ... (only orthogonality observed, for lower alpha I need to add the more diverging normals back)
     * plus bilateral filtering
     * plus mean-shift
     * enforce the orthogonality 
   * should I try?
        * how much the orthogonality helps
        * synthetic data...

 * saving tentatives locations and #inliers against GT
   * problem with #inliers against GT - errors too low: tried Sampson error - demo?
   * tried "simple weighing": see the spreadsheet 
 
 * TODO rectified patches without keypoints      
