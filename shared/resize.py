from math import fabs, log, floor, ceil
import torch

# (multiple of 32 so that it best preserves the ratio x 512)
def resize_multiple_32_times_512(orig_dimensions_h_w):

    (h, w) = orig_dimensions_h_w

    if w < 512:
        raise Exception("Unexpected - not implemented")

    ratio = float(h) / w # float ?
    ideal_height = 512 * ratio
    multiple_32 = ideal_height / 32 # float

    ceil_score = abs(log(multiple_32) - log(ceil(multiple_32)))
    floor_score = abs(log(multiple_32) - log(floor(multiple_32)))

    if ceil_score < floor_score:
        ret = (int(ceil(multiple_32)) * 32, 512)
    else:
        ret = (int(floor(multiple_32)) * 32, 512)

    return ret


def upsample(depth_data, height, width):

    depth_data = depth_data.view(1, 1, depth_data.shape[0], depth_data.shape[1])
    upsampling = torch.nn.Upsample(size=(height, width), mode='bilinear')
    depth_data = upsampling(depth_data)
