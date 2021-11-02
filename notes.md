

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


# notes from 10/22/2021

## running on GPU on cluster

 * 80-fold speed up for the clustering experiment (+ caching scheme for sky mask and normals)
 * https://github.com/pytorch/pytorch/issues/41306

## last experiments on clustering

 * real robust optimum actually may be around 25 degrees (35 degrees kind of enforces orthogonality) 

## megadepth ds

 * overview of the ds (one "scene") / many scenes
 * depth estimation demo
   * how to measure that?
 * cluster normals orthogonality demo   
 * antipodal points demo

## next steps
 
 * finally start experiments with matching
   * maybe again try some combinatorial search for the rectification angle
   * many other options
   * AI: let's start with the unrectified kpts added
 * honestly more/better ds would be helpful
   * may again try some synthetic data
 * plane normals idea
 * plane patches - bilateral filtering
 * AI: use kornia for computations on GPU 
 * AI: try to use other depths gt: bookmarked
 * AI: check the antipodal points logic 


# notes from 10/29/2021

## experiments

* the last improvements in normal estimation improved the overall performance!
  * but adding unrectified kpts and handling the antipodal points harmed the performance
  * still interested in seeing how (exactly) the normals accuracy influence the overall performance
    * correlation diff(right angle) <-> error in relative rotation
    * recompute the normals for contiguous patches
    * combinatorial search  
    * enforcing the orthogonality
  * the same goes for the area of the patches 
    * bilinear filtering      
    * grow - shrink the patch
   
## what to try next

* smaller rotation for the rectification
  * half the rotation
  * search for R so that ||R(n_1_i) - n_2_i|| is minimized
    * then use half the rotation for the rectification 
    * I can observe how the final/GT R is consistent with ||R(n_1_i) - n_2_i||
      * this is another way to check the normals estimation btw. (if I have GT R)
      * this also may suggest some kind of reestimation loop, etc.
    * plus constraints on R (Rz = I)

* again try some DS with non rectangular surfaces 
* Q: parameters for SIFT (n_features...) / possibly other feature extractor
   
## The thesis update(s)

* update the description of the techniques
* clean up the presentation (graphs), template
* how to consult it? (diff?)

* special section(s) for contribution

## Workflow 

* smoother workflow 
  * generating better tabular data / graphs 
  * better archiving, etc. 

## minutes

* (try 0.9 for all unrectified)
  * but maybe better: https://github.com/ubc-vision/image-matching-benchmark/blob/master/methods/feature_matching/nn.py#L149
* SIFT - try 8K
* try HardNet from https://github.com/kornia/kornia-examples/blob/master/MKD_TFeat_descriptors_in_kornia.ipynb and from below


```
def get_local_descriptors(img, cv2_sift_kpts, kornia_descriptor):
  if len(cv2_sift_kpts)==0:
    return np.array([])
```
```
  #We will not train anything, so let's save time and memory by no_grad()
  with torch.no_grad():
    kornia_descriptor.eval()
    timg = K.color.rgb_to_grayscale(K.image_to_tensor(img, False).float()/255.)
    lafs = laf_from_opencv_SIFT_kpts(cv2_sift_kpts)
    patches = KF.extract_patches_from_pyramid(timg,lafs, 32)
    B, N, CH, H, W = patches.size()
    # Descriptor accep
    B, N, CH, H, W = patches.size()
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :) 
    descs = kornia_descriptor(patches.view(B * N, CH, H, W)).view(B * N, -1)
  return descs.detach().cpu().numpy()
descs1 = get_local_descriptors(img1, kps1, descriptor)
```

