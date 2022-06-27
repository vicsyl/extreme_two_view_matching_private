import torch

from pipeline import *
from dense_affnet_feature import DenseAffnetFeature
import kornia as K


def read_img(img_file_path):
    img = cv.imread(img_file_path, None)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # plt.figure(figsize=(9, 9))
    # plt.title(img_file_path)
    # plt.imshow(img)
    # plt.show(block=False)
    return img


def process_files(file_paths):

    # local_config = CartesianConfig.parse_file_for_single_config("standalone_laffs_config.txt")

    dense_affnet_feature = DenseAffnetFeature(device=torch.device('cpu'))

    for file_path in file_paths:

        img = read_img(file_path)
        img_t = K.image_to_tensor(img, False).float() / 255.

        mask = torch.ones_like(img_t, dtype=bool)
        mask[:, :, :500] = 0

        laffs, responses, descs = dense_affnet_feature.forward(img_t, mask)
        print("{} done".format(file_path))


if __name__ == "__main__":

    file_names = ["original_dataset/scene1/images/frame_0000001555_4.jpg",
                  "original_dataset/scene1/images/frame_0000000945_4.jpg"]

    process_files(file_names)
