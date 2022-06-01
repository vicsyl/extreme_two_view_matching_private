from pipeline import *


def read_img(img_file_path):
    img = cv.imread(img_file_path, None)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(9, 9))
    plt.title(img_file_path)
    plt.imshow(img)
    # plt.show(block=False)
    return img


def process_files(file_paths):

    config = CartesianConfig.parse_file_for_single_config("standalone_laffs_config.txt")

    feature_descriptor = Pipeline.setup_descriptor_static(config, device=torch.device("cpu"))
    assert isinstance(feature_descriptor, HardNetDescriptor), "rectify_affine_affnet on, but without HardNet descriptor"
    dense_affnet = DenseAffNet(True)

    for file_path in file_paths:

        img = read_img(file_path)
        img_data = affnet_clustering(img, file_path, dense_affnet, config, upsample_early=True)

        kpts_struct = affnet_rectify(file_path,
                                     feature_descriptor,
                                     img_data,
                                     config)


if __name__ == "__main__":

    file_names = ["original_dataset/scene1/images/frame_0000001555_4.jpg",
                  "original_dataset/scene1/images/frame_0000000945_4.jpg"]

    process_files(file_names)
