import argparse
import cv2 as cv
import matplotlib.pyplot as plt

from utils import parse_device, use_cuda_from_device
from dense_affnet_feature import DenseAffnetFeature, get_default_config
from kornia_moons.feature import visualize_LAF
from sky_filter import *
from kornia_utils import show_torch_img
from img_utils import numpy_to_torch_img


def read_img(img_file_path):
    img = cv.imread(img_file_path, None)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(9, 9))
    plt.title(img_file_path)
    plt.imshow(img)
    plt.show(block=False)
    return img


def get_non_sky_mask(img_np, use_cuda):
    img_torch_for_sky = get_torch_image(img_np, img_np.shape[0], img_np.shape[1], use_cuda=use_cuda)
    mask = get_non_sky_mask_torch(img_torch_for_sky, use_cuda=use_cuda, cpu=False)
    mask = mask[None, None].repeat(1, 3, 1, 1)
    return mask


def process_files(file_paths):

    config = get_default_config()
    # many more visualizations:
    # config["show_affnet"] = True
    # this will only detect every n-th feature (default=None, which is effectively 1)
    config["affnet_hard_net_filter"] = 50
    # subsumpling ratio over what dense affnet already does (default=None, which is effectively 1)
    config["affnet_dense_affnet_filter"] = 2
    config["show_dense_affnet_components"] = True

    parser = argparse.ArgumentParser(prog='standalone_laffs')
    parser.add_argument('--device', default='cpu', help='torch device')
    args = parser.parse_args()
    device = args.device
    print("device = {}".format(device))
    config["device"] = device

    device = parse_device(config)
    use_cuda = use_cuda_from_device(device)

    dense_affnet_feature = DenseAffnetFeature(device=device, config=config)

    for file_path in file_paths:

        img = read_img(file_path)
        img_t = numpy_to_torch_img(img)
        non_sky_mask = get_non_sky_mask(img, use_cuda=use_cuda)
        show_torch_img(non_sky_mask, "not downsampled sky mask")

        laffs, responses, descs = dense_affnet_feature.forward(img_t, non_sky_mask)

        laffs_scaled = laffs * responses[:, :, None]
        laffs_scaled[:, :, :, 2] = laffs[:, :, :, 2]
        every_other = 5
        visualize_LAF(img_t, laffs_scaled[:, ::every_other], figsize=(9, 9))

        print("{} done".format(file_path))

    print("All done!")


if __name__ == "__main__":

    file_names = ["original_dataset/scene1/images/frame_0000001555_4.jpg",
                  "original_dataset/scene1/images/frame_0000000945_4.jpg"]

    process_files(file_names)
