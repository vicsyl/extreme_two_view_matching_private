import cv2 as cv
import torch
from evaluation import ImageData
from affnet import affnet_rectify
from sky_filter import get_nonsky_mask
from affnet import winning_centers, decompose_lin_maps_lambda_psi_t_phi
from dense_affnet import *
from scene_info import *
from opt_covering import *
from connected_components import *

from rectification import possibly_upsample_normals
from hard_net_descriptor import HardNetDescriptor


def upsample(data_in, to_size):
    """
    :param data_in: torch.Tensor(B, C, H, W)
    :param to_size: (H2, W2)
    :return: torch.Tensor(B, C, H2, W2)
    """
    assert data_in.shape[2] < to_size[0]

    upsampling = torch.nn.Upsample(size=to_size, mode='nearest')
    data_in = upsampling(data_in)
    return data_in


def get_default_hardnet(filter=20):

    sift = cv.SIFT_create(None, 3, 0.04, 10, 1.6)
    hardnet_descriptor = HardNetDescriptor(sift,
                                           compute_laffs=False,
                                           filter=filter,
                                           device=torch.device("cpu"))
    return hardnet_descriptor


def read_img(img_path):

    img = cv.imread(img_path, None)
    if img is None:
        raise ValueError("img not found at {}".format(img_path))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(9, 9))
    plt.title(img_path.split("/")[-1])
    plt.imshow(img)
    show_or_close(True)

    return img


def process(img_path, dense_affnet):
    img = read_img(img_path)
    timg = K.image_to_tensor(img, False).float() / 255.
    timg = K.color.bgr_to_grayscale(timg)

    with torch.no_grad():
        aff_features, all_features_flat = dense_affnet(timg)

        # # aff_features = aff_features_flat.reshape(-1, 4)
        # # aff_features = aff_features.reshape(BB,' HH, WW, 4)
        # # aff_features = aff_features.permute(0, 3, 1, 2)
        # # aff_features_upsampled = upsample(aff_features, img.shape[:2])
        #
        # for i in range(2):
        #     for j in range(2):
        #         t_i = K.tensor_to_image(aff_features[:, :, i, j])
        #         print('s1:{}'.format(t_i.shape))
        #         plt.imshow(t_i)
        #         plt.show()

        non_sky_mask = get_nonsky_mask(img, aff_features.shape[0], aff_features.shape[1], use_cuda=False)
        non_sky_mask = torch.from_numpy(non_sky_mask).reshape(-1, 1)[:, 0]

        covering: CoveringParams = CoveringParams.dense_covering_1_7()

        conf = {"show_affnet": True,
                "affnet_covering_fraction_th": 0.97,
                "affnet_covering_max_iter": 100,
                "invert_first": True,
                "affnet_include_all_from_identity": True,
                "affnet_no_clustering": False,
                "affnet_covering_type": "dense_cover",
                }

        _, _, ts1, phis1 = decompose_lin_maps_lambda_psi_t_phi(aff_features, asserts=False)
        # this doesn't work

        H, W = ts1.shape[:2]

        HF, WF = aff_features.shape[:2]
        assert HF == H
        assert WF == W

        ts1_flat = ts1.reshape(1, -1)
        phis1_flat = phis1.reshape(1, -1)
        ts_phis, cover_idx = winning_centers(covering, ts1_flat, phis1_flat, conf, return_cover_idxs=True, valid_px_mask=non_sky_mask)

        cover_idx = cover_idx.reshape(H, W)
        for i in range(len(ts_phis) + 1):
            plt.imshow(cover_idx == i)
            plt.show()

        upsample_early = True
        if upsample_early:
            cover_idx = possibly_upsample_normals(img, cover_idx.numpy())
            cover_idx = torch.from_numpy(cover_idx)

        all_valid = range(1, len(ts_phis))
        components_indices, valid_components_dict = get_connected_components(cover_idx, all_valid, show=True)

        if not upsample_early:
            assert np.all(components_indices < 256), "could not retype to np.uint8"
            components_indices = components_indices.astype(dtype=np.uint8)
            components_indices = possibly_upsample_normals(img, components_indices)
            components_indices = components_indices.astype(dtype=np.uint32)

        img_data = ImageData(img=img,
                         real_K=None,
                         key_points=None,
                         descriptions=None,
                         normals=None,
                         components_indices=components_indices,
                         valid_components_dict=valid_components_dict)

        hardnet_descriptor = get_default_hardnet()
        conf["affnet_covering_max_iter"] = 1

        affnet_rectify(img_path.split("/")[-1],
                       hardnet_descriptor,
                       img_data,
                       conf,
                       device=torch.device('cpu'),
                       params_key="",
                       stats_map={})

        # affnet_rectify (cfg[max_iter == 1] + compare(how - visually?))
        # img_data - check in debug
        # productionize (git; pipeline, test, clean & git, run; run baseline)

        # _, _, ts, phis = decompose_lin_maps_lambda_psi_t_phi(all_features_flat, asserts=False)
        # ts_phis2 = winning_centers(covering, ts, phis, conf)

        # print("eq1: {}".format(torch.equal(ts, ts1_flat)))
        # print("eq2: {}".format(torch.equal(phis, phis1_flat)))
        # aff_features_f2 = aff_features.reshape(1, -1)
        # print("eq3: {}".format(torch.equal(aff_features_f2, all_features_flat)))
        #
        print("Done")


def main():

    dense_affnet = DenseAffNet(True)
    scene_info = SceneInfo.read_scene("scene1")
    file_names, _ = scene_info.get_megadepth_file_names_and_dir(None, None)
    for idx, depth_data_file_name in enumerate(file_names[:2]):
        process(scene_info.get_img_file_path(depth_data_file_name[:-4]), dense_affnet)


if __name__ == "__main__":
    main()

