# Two-view matching of image containing planar surface exploiting monodepth estimation

... this is the whole title for the diploma thesis 

* based on a paper @ https://arxiv.org/pdf/2008.09497.pdf

## How to run it

* the easiest is to run the whole pipeline (pipeline.py)
* the parameters are set programmatically in the code; I think they are quite self-descriptive and currently set reasonably
* the inputs are the depth maps from megadepth - expected under ./depth_data/mega_depth/scene1
  * one can use sample data from https://github.com/vicsyl/extreme_two_view_matching_data_samples/tree/master/megadepth_data
* the pipeline will save results into ./work/scene1/matching/pipeline_with_rectification and 
./work/scene1/normals/simple_diff_mask respectively

## General overview

### Depth estimating CNNs

#### MegaDepth

* https://github.com/zhengqili/MegaDepth
* preprocessing - rescaling the images to have a maximum dimension of 512, with the other
dimension chosen as the multiple of 32 that best preserves the original aspect ratio (as in the original paper)
* poor estimation of sky depth - however this also means that the normals are prety much random so sky gets filtered out pretty reliably in the end

#### MonoDepth2

* https://github.com/nianticlabs/monodepth2
* rescales the image exactly according to the model (will be using 640x192 as it's closest to the resolution used in MegaDepth)
* is supposed to be estimating sky better, but ... let's see
* doesn't seem that good from the pictures
* min/max depth bounds (not very tight)
* (TODO) depth data generated, but they are still to be tried in the rest of the processing 

### Normals from depth

* (PROBLEM, parameter to be tuned) the depth given up to the scale (at least for MegaDepth), which greatly affects the normals (see depth_factor in depth_to_normals.py/compute_normals_simple_diff_convolution)
* (QUESTION/REQUEST) it would help me to have some scenes with known normals of the dominating planes. The normals computed I see are ok-ish, but I am not sure (e.g. cannot judge from the rectification, which itself may have some discrepancies) 

#### Normals through simple differential conv mask

* (IMPROVEMENT, TODO) - adjustment counting with the exact direction of the projecting rays not done 
* similar are normals through sobel conv mask

#### Normals through fitting a plane

* (PROBLEM, TODO) - seems to be to slow, but I can still try torch.unfold to parallelize it and speed it up

### Processing of the normals

#### Clustering normals

* spherical k-means used for the clustering
* filtering the normals under sharp angles (over the threshold of 80 degrees according to the original paper) seems handle the sky pretty neatly. 
* I think there is a lot of room for a) improvement b) parameter tuning c) possibly a speed up
* (PROBLEM, TODO) I think it would improve the results if I can estimate the number of the clusters (i.e. dominating planes). 
So far I labeled some of the scenes manually with the expected number of dominating planes which seem to help the results

#### Connected components

...are found to 
* define regions that can be rectified all at once to improve efficiency of the rectification
* filter out components which are too small and most likely correspond to some clutter  

### Rectification

* (PROBLEM, needs attention) probably due to some rounding error small fraction of the keypoints are back mapped to invalid positions 
(it may also be the descriptor finding keypoints at/beyond the border of the original img when mapped with the rectification homography)
* now done so that really the smallest possible area is mapped with the rectifying homography

### Matching

* pretty straight-forward
* only SIFT is used so far as a descriptor 

### Evaluation

* (TODO) not part of the whole pipeline, yet
* adopted the evaluation of pose (essential matrix) estimation from https://github.com/ducha-aiki/ransac-tutorial-2020-data
* also, it seems interesting to me that the papers only mentions the rotation error, not the translation error 
* I would still like to look at different metrics (e.g. number of correctly matched keypoints - according to the ground truth)

## Speed performance

* depth estimation by megadepth: around 7 secs.(!) - this is quite surprising as it takes monodepth much less time   
* the whole processing (depth -> normals -> clusters and components -> rectification -> matching) takes about 13 seconds 
per an image pair
* reading the scene info (once per execution) takes about 4-5 seconds 
