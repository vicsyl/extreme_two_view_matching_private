

## questions:

 * how to improve the process?
   * i am tempted to rewrite it all together
     * clean up the code
     * gpu
     * definitely i think i can rewrite at least some parts
   * have a process for quick experiments and their acceptance/rejection   
    

# results / done:

 * multiple params for the clustering:
   * observing how close the angle between 2 biggest planes are to the right angle - see the spreadsheets
   * ongoing research ... (only orthogonality observed, for lower alpha i need to add the more diverging normals back)
     * plus bilateral filtering
     * plus mean-shift
     * enforce the orthogonality 
   * should i try?
        * how much the orthogonality helps
        * synthetic data...

 * saving tentatives locations and #inliers against gt
   * problem with #inliers against gt - errors too low: tried sampson error - demo?
   * tried "simple weighing": see the spreadsheet 
 
 * todo rectified patches without keypoints      




# notes from 10/15/2021

## experiments

 * local - need to repeat last bigger experiment
 * the average absolute diff from the right angle should be ~4 deg (was 6)
 * params:
    * big bins (35 degrees)
    * sing. values quantil (0.8 - 1.0)
      * tried s2/s3 rather than s3/depth
    * simply taking a mean (vs. mean shift)
    * sigma for svd weighting low (0.8)
    * bilateral filter doesn't seem to help correct the normals
      * will help later augment the patches
    * simple x counterintuitive
    * i now let 3 planes to be detected (this may need a little tuning for the chosen params values / for all params values)  
    
## mean shift theory

## antipodal points - handling 

 * not only to clone the logic, but also to handle better planes along z-axis 

 * flip if z > 0 - this is ok so that there are no normals with high z>0 coord
      
        where = torch.where(normals[:, :, 2] > 0)        
        normals[where[0], where[1]] = -normals[where[0], where[1]]
     
 * now
     
        diffs = n_centers - normals
     
 * ...becomes (this will only affect bins with z close to zero)
     
        diffs[0] = n_centers - normals
        diffs[1] = n_centers + normals    
        diffs = torch.min(diffs, dim=0)
   
 * ... but also need to check 

        def is_distance_ok(new_center, threshold):
            for cluster_center in cluster_centers:
                # diff = new_center - cluster_center
                # diff_norm = torch.norm(diff)
                diff_new = torch.vstack((new_center - cluster_center, new_center + cluster_center))
                diff_norm = torch.norm(diff_new, dim=1).min()
                if diff_norm < threshold:
                    return false
            return true

## next steps

 * run last round of experiments
 * read the ds to be able to try the pipeline on it (normals orthogonal? improves matching?)
 * pick the best clustering params as default
 * try some ideas on matching (union rectified and unrectified kpts)
