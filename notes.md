

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

# notes from 11/5/2021


## experiments

* made the pipeline more flexible 
* tried Hardnet, n features (SIFT_create), fginn
  * not ready but doesn't seem to show major improvement
  * Hardnet not ready
  * fginn (1st geometrically inconsistent nearest neighbor ratio)
    * params for 
      * num_nn = 2, fginn_spatial_th = 100, ratio_th = 0.5
      * try these: num_nn = 2, fginn_spatial_th = 10/15, ratio_th = 0.85
    * exceptions for fginn
      * too few tentatives, 5 inliers => multiple E? E is 3x6 ? (multiple Es)
  * n_features = 8000 slightly better in medium difficulties (how does it even work?)
* tried fginn with all unrectified kpts
  * change params / observe some metrics (correspondence between unrectified (on some patches) and rectified )
    
## rotation estimation from normals 

* code 
* estimation performance 
* TBD
  * rectify half way through
  * constraints
  * look closely at the estimation performance
* theory
  * continuous function on compact set -> attains minimum 
  * how many local minima are there (more of them may be interesting)  


## what's next

* continue working on rotation estimation from normals
* parameters for fginn
* observe new metrics?
* old stuff (bilinear filtering, recomputing the normal of the patch, etc...) 

## minutes 

* n_features = None probably OK, other params: edge threshold for SIFT.... higher value is more permissible x contrast threshold the other way around
* try these: num_nn = 2, fginn_spatial_th = 10/15, ratio_th = 0.85
* rarely I can get multiple Es (multiple E? E is 3x6 ? (multiple Es))


# notes from 11/12/2021

## experiments

* fginn with better params with all keypoints (sent in the email), still worse than the baseline
  * check the inlier ratio vs. number of tentatives
  * how many keypoints added / increase in the # of keypoints    
  * keypoints analysis
    *  patch / unrectified tentatives(/inliers) correspondence
    *  how many inliers wrt GT 
    *  how does a kpt improve the correspondence when rectified / shadows / is shadowed by unrectified correspondence
  * how the fginn works 
    * ( UPDATE: no diff in center of gravity of (xdescs)) - UPDATE: it just removes some matches from near location 
    * why should it work in this use case (?)
    
* hardnet
  * CUDA: fast (recomputed without rectification), BUT sometimes out of memory
     * memory leak? - probably not  
     * may even dynamically programmatically change CUDA core(?) - https://kornia.readthedocs.io/en/latest/utils.html#kornia.utils.batched_forward
  * !why do the results make sense actually? (x brisk (opencv) x superpoint()) - 
    https://github.com/magicleap/SuperPointPretrainedNetwork , 
    https://github.com/ubc-vision/image-matching-benchmark-baselines
    https://github.com/ubc-vision/image-matching-benchmark-baselines/blob/master/run_superpoint.sh
    https://github.com/ubc-vision/image-matching-benchmark-baselines/blob/master/third_party/superpoint_forked/superpoint.py#L550
    
  * should I compare with our old pipeline (with hardnet), then? - ?the (slight) improvement in estimation brings big improvement with hardnet?

* rotations
  
  * kept the parametrization via rotation vector
    * I watch the objective function
    * minor problems with the numerical minimizers though (global minima, rot_v + 2.pi.rot_v / |||rot_v||)
    * I can check with approaches using GT for now 

  a) by dR: rotate by 1/2 of estimated dR (or GT dR / or dR|z=0)
    * performance is basically the same as without rectification - !! double check
    * GT - very straight-forward, should it work? - does it mean the approach is just fundamentally worse than the normalization by orthonormal projection?
      * having GT - if I rotate the image, do I have just scaling correspondences (per corresponding planes)?   
  
  b) by alpha * rectification rotation
    * not very convincing either
    * results interesting though (asymmetry wrt alpha1/alpha2)  
  
  * lots of possibilities (R|z=0 R|y, R|min=0), relative rotation <-> orthogonality to z, etc.
    * repeating patterns problem (R|y)
    * will try to analyze some metrics on the kpts data  
  
## what's next 

* rotations 
  * more experiments / analysis
  * why doesn't the rotation via GT dR / 2 work?
    * normals estimation (-> GT) -> rectification: the last step (the highest level) of the method
  * many possibilities
    * meta idea -> estimated GT and reiterate (i.e. I now know the plane correspondences)
* asift (compared to hard net / or combined with(?))
* ! another idea: jupyter notebook  
* old stuff

## questions

* the numbers from the original paper: with their own CNN? They achieve almost 100% on first 2 difficulties(!) - ASK (to all authors)!! - on the entire dataset 
* numbers with hardnet almost the same (somewhere better, somewhere worse) 

## Action items

* Hardnet
  * CUDA + memory: https://kornia.readthedocs.io/en/latest/utils.html#kornia.utils.batched_forward
  * run with the previous (worse version) of the pipeline
  * results conf. brisk (opencv) x superpoint - try these 
    * SuperPoint 
      * https://github.com/magicleap/SuperPointPretrainedNetwork 
      * https://github.com/ubc-vision/image-matching-benchmark-baselines
      * https://github.com/ubc-vision/image-matching-benchmark-baselines/blob/master/run_superpoint.sh
      * https://github.com/ubc-vision/image-matching-benchmark-baselines/blob/master/third_party/superpoint_forked/superpoint.py#L550
* rotation by GT/2 
  * ! double check
* try to analyze the perf of the alpha rotation on different pairs
* metrics.... what happens in general
* !new idea - see the email
* the numbers from the original paper they achieve almost 100% on first 2 difficulties - even without rectification (!) (scene1 / the entire dataset) 
  * ASK (to all authors)!!  

# notes from 11/19/2021

* no answer from authors of the original paper ;-( 

## experiments

* BRISK and SuperPoint (retained the copyright notice)
  * rectified SuperPoint better
  * otherwise, it corresponds with original observations  
* HardNet on old version (batching fixed the problem with CUDA)
  * new version a little better
  * actually back-ported (this time easy), but in the future I think I should just set the params to original values  

## AffNet 

* the main idea: 
  * AffNet affine info (+depth) -> rectify, detect again
    * as apposed to Rodriquez approach 
      * a) only parts of the image are rectified (just by one transformation per each part)
      * b) depth info can be used 
    * the expected outcome: faster (computations time for not needed transforms saved), more accurate (assuming the rectification transform can be more accurately estimated)
    * the accuracies in the paper are IMHO all close to 100% and the main contribution seems to be the speed up (e.g. compared to ASIFT)
* notes:
  * in-place description recomputation  
    * even for different normals at each kpt the rectification (memory, time) could be saved
    * even though AffNet works with affine transforms, the descs are computed by another subpart (CNN) - most likely the recomputation is not possible here 
  * I am skeptical, because 
    * a) AffNet kpts may be too sparse - maybe not
    * b) the normals estimation based on Affnet would generally need to improve on the estimation base on the depth
    * c) AffNet gives the normalizing affine map - may not be to fronto-parallel and may be different from a feature to feature...   
    + let's first compare the normal estimates based on AffNet and megadepth - normals/clusters..
    

## Investigation

* observed metrics (plane <-> plane correspondence)
  * use less restrictive thresholds (not 0.5 pixels)

* many inliers
  * most of the errors for diff (0, 1)
  * seems to depend on
    * CUDA/vs. CPU(!!) - non-determinism, BUT it seems quite reproducible given the setting
    * version of the library 
    * differs from computation to computation (I didn't see it)      
    * better result on version 1 may not mean anything (may fail on other)
  * seems to be low-hanging fruit (admittedly for rather easy difficulties only)     
  * why does it go wrong (repeating patterns?) - many inliers wrt. GT AND wrt the incorrectly estimated dR 
  * prefiltering matches on corresponding planes seem not to work (not too many matches to be filtered out, the problem can appear after the filtering)  
  * use the parallax (induced by the estimated plane should be ~ 0) / otherwise use the depth

* not many inliers
  * still many keypoints - will try the filtering before the ratio filtering
  * parallax 
  * ransac vs. plane correspondence
    * cannot try ((1, 2), (1, 1)), only ((1, 2), (1, 2)) or ((1, 1), (1, 1))
      * 1/8 of possibilities with 1/2 kpts  
    * deep inside ransac? - NO: run RANSAC multiple times

* generally diff (sigma) & diff (rho) may be used as in AdaLam 
  * needs to be homography and not affine
  * needs to cope with the unknown scale  

* would like to see what happens with all_rectified = True (the kpts will clash)


## what's next 

* more experiments around the rectification and matching
* some things may be done/needed on the lower level - hopefully it won't break the existing observations
* when would it be the time to wrap it up? What should I do in terms of the computations 
  * (other scenes, other DSs, more combinations of params, new kpts detectors, etc..)
    
## Action items

* AffNet
* filtering based on the plane correspondences 
  * tentatives, RANSAC


* https://hal.archives-ouvertes.fr/hal-02156259v2/document
* https://math.stackexchange.com/questions/861674/decompose-a-2d-arbitrary-transform-into-only-scaling-and-rotation


# notes from 11/26/2021

## the space of tilts ~ AffNet

* approach taken: cluster the decompositions of affine maps of AffNet features (space of tilts) based on the clustering
* per detected connected component (not only normal)
* comparison with the normal doesn't look convincing: it's important that the affine maps themselves are aligned
* demo  
* still finishing the whole pipeline
  
* generally I think that affine maps may be better than the homographies
  * less memory, faster
  * aligned with the (apparently fruitful) approach taken by Rodriguez and others ("IMAS methods")  

* covering of the space of tilts per component
  * a) same approach as in Rodriguez - fixed covering 
    * (the speedup without the depth/normals info very impressive)
  * b) compute the optimal covering per component (or maybe combined with c))
  * c) just use the mean (there needs to be just one cluster) - I am trying now
    
  * what about the rest of the image  

* what if clustering is applied based on affnet features
* TODO gaussian blur
* https://github.com/opencv/opencv/blob/master/samples/python/asift.py#L59


## submission deadline

* 4th January (5 1/2 weeks)
* I would like to still keep experimenting for some time 
* will try to work on the text (of what I already have)


# notes from 11/26/2021

## Experiments

* RootSIFT + settings they had (does it mean I )
* (Affnet) 

## Demo on AffNet

* demo
* color(v(x, y)) = color(u(A(x,y)) (coordinate transformation) (ASIFT, AffNet)
* x_v, y_v <- B(x, y)  (img transformation) 
* A = inverse(B)
* imagine A = 2 * eye(2)

## Questions

* Homography => how to compute the direction for the gaussian blur
  * https://cmp.felk.cvut.cz/~chum/papers/Chum-CVIU05.pdf
  * http://people.ciirc.cvut.cz/~hlavac/TeachPresEn/17CompVision3D/11Geom1camera.pdf
  * compare if the norm affine maps would change along with the changing direction of "maximal shrink"     

* RGB vs BGR

## Plan

* need to start writing the content
  * let me start with AffNet

* what should I still compute (DS, settings)    

* how frequently should we sync (online, offline)


# notes from 12/17/2021

# experiments

* rectification driven via AffNet affine maps - results
  * naive covering the space of tilts via one mean per component
  * covering inspired by Rodriquez - resembling also the voting during the clustering
    * demo    
  * how to improve the Affnet (driven rectification via affine maps) even more?
    * keep top 400 sorted by ratio 
    * there are many keypoints found (many thousands), but only few tentatives - different matcher parameters (distance ratio threshold - currently 0.85)
  * there is an obvious accuracy / resources (time, memory) trade-off - for these coverings the memory footprint / comp. time should not be that bad    

* plan
  * try on the rest of the DS
  * try on different DS - st. peters, EVD,...
  * try with estimated K
  * Q: can most of the measurements done just by scene1 and the whole data sets on few standard settings? 
  * A: EVD compute homography

- comparison with Rodriques
- comparison with Torsten
- ICPR - dates? - ping ... 

  * shouldn't take long to try out the gaussian blur (I will use the decomposition of H = H_p @ Aff) 
  * with minor modifications I can actually try the exact method by Rodriquez - to comparison
  * complete (and rework) the thesis itself


# notes from 12/22/2021

# experiments

* in progress
  * the whole original DS
  * Rodriquez approach on scene 1 - in progress

# thesis content

* used the official template - questions mostly regarding the language (will discuss with department)
* it needs reworking really
* roughly divided into 2 parts: Toft approach and HardNet 


# questions

* Sharing on github? With source? (/export pdf to st editable)
  
* Wording - copying from the original paper 
  * my own style / notation consistency 
  * the source is listed in references

* Structure 
  * not sure whether the structure logically makes sense to a (first) reader 
  * parts may be needed to be swapped/amended
  
* format - I don't really like my graphs/pseudocode, etc.
  * will improve it gradually
  * can I use img taken from other papers? (I may come up with my own at the very end)

# plan 
* annotate the work with notes and TODOs so that it can be revised
* finish the new section around ASIFT/Space of tilts/HardNet
* shall we sync on Monday?


# notes from 12/28/2021

# experiments

* in progress
  * the whole original DS
  * Rodriquez approach on scene 1 - in progress

# thesis content

* used the official template, imported to overleaf
* roughly divided into 2 parts: Toft approach and HardNet
* reworking the first part   
* comments in orange boxes

# questions

* copying from the original paper 
  * my own style / notation consistency 
  * the source is listed in references

* could the chapter 2 be revised even if incomplete? 
  * for the structure
  * many things (the basic assumptions, steps of the pipeline, ...) are repeated
  * the formalization in the equations? 



* plan to the document

* list of contributions 
* experiments
* graphs - line style, don't use yellow, 


# plan for 12/29/2021

* check the experiments, plan next one - sync with Dmytro 
* content 
  * list of contributions - OK
  * notes (+plan) for Dmytro  
  * results (optimize for throughput) (+ experiments)
  * images (optimize for throughput)
* email Dmytro  

## high level plan
* 12/29: 2nd chapter + results : 20+ pages)  
* 12/30-31 - 3rd chaper ~30 pages
* 1/1 - 3/1 - 40 pages
* 4/1 - check it!!!


# plan for the experiments

## Finished (soon)
* final variants of Rodriguez on scene 1 - will finish tonight 
* toft DS complete on unrectified: done in 2 hours

## To compute

* toft DS complete (computationally expensive)
  * best rectified version for SIFT
  * best Rodriguez improvment
  * (standard Rodriquez)
  
* other
  * do not use google urban
  * estimated K 
  * other DS ?

* what will bring value   


# plan/notes for 12/30/2021

## Plan 
* reconcile the results (including the "low level results")
* run what is planned 
* add imgs
* continue with the results section
* revise the plan for the content (#pages)

# plan/notes for 12/31/2021
* I restarted the experiments on preprocessing (angles between estimated normals), should be ready tomorrow morning
* handle antipodal points - ....
* k-means => simply didn't work - I can mention it like that
  
* algoritm 1 -> how can it be improved? 
* algoritm 2 -> I will rewrite to equations (I think it will be better) 

Go to the document....
* estimate K on Toft DS?


https://openreview.net/pdf?id=TVHS5Y4dNvM
https://carbon.now.sh/
https://arxiv.org/pdf/1711.07064.pdf

pseudo_code:


     
https://github.com/ducha-aiki/ransac-tutorial-2020-data/blob/52810309d8341d538e24a13577c44ae2b4a5ec77/metrics.py#L5
https://github.com/ducha-aiki/ransac-tutorial-2020-data/blob/master/create_H_submission.py






  
