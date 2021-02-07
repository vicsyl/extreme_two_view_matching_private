# SVP Project

## Depth estimating CNNs

### MegaDepth

* https://github.com/zhengqili/MegaDepth
* preprocessing: rescaling the images to have a maximum dimension of 512, with the other
dimension chosen as the multiple of 32 that best preserves the original aspect ratio (as in the original paper)
 (only now!!!)
* poor estimation of sky depth
* explore the range of the depth values!!! 


### MonoDepth2

* https://github.com/nianticlabs/monodepth2
* rescales the image exactly according to the model (will be using 640x192 as it's closest to the resolution used in MegaDepth)
* is supposed to be estimating sky better, but ... let's see
* doesn't seem that good from the pictures
* min/max depth bounds (not tight!!!)


## Normals from depth

* factor (range) of the depths!!!!
* projection - adjustment for the dispersion of the rays according to the depth 
