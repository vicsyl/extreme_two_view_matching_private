import cv2 as cv
import matplotlib.pyplot as plt
import torch
from evaluation import ImageData
from affnet import affnet_rectify
from sky_filter import get_nonsky_mask_torch
from affnet import winning_centers, decompose_lin_maps_lambda_psi_t_phi
from dense_affnet import *
from scene_info import *
from opt_covering import *
from connected_components import *
from depth_to_normals import show_sky_mask
import kornia.feature as KF
from config import CartesianConfig

import sys

from rectification import possibly_upsample_normals
from hard_net_descriptor import HardNetDescriptor


def affnet_coords(dims_2d):
    """
    :param dims_2d: dimension tuple (H, W)
    :return: torch.Tensor (2, H, W)
    """
    h_m = torch.linspace(0, dims_2d[0] - 1, dims_2d[0]) * 4 + 14
    w_m = torch.linspace(0, dims_2d[1] - 1, dims_2d[1]) * 4 + 14
    mesh = torch.meshgrid(h_m, w_m)
    mesh_tensor = torch.vstack((mesh[0][None], mesh[1][None]))
    return mesh_tensor


def torch_upsample_factor(data_2d, factor):
    data_2d = data_2d[None, None]
    upsample = nn.Upsample(scale_factor=factor, mode='nearest')
    upsampled = upsample(data_2d)[0, 0]
    return upsampled


def affnet_upsample(data_2d):

    data_2d = data_2d[None, None]
    upsample = nn.Upsample(scale_factor=4, mode='nearest')
    upsampled = upsample(data_2d)
    replication_pad = nn.ReplicationPad2d(15)
    padded = replication_pad(upsampled)
    padded_centered = padded[0, 0, 2:, 2:]

    def assert_dim(dim_in, dim_out):
        assert (dim_out / 4) - 7 == dim_in
    assert_dim(data_2d.shape[2], padded_centered.shape[0])
    assert_dim(data_2d.shape[3], padded_centered.shape[1])

    return padded_centered


def show_affnet_features(aff_features, conf):

    if not conf.get(CartesianConfig.affnet_show_dense_affnet, "False"):
        return

    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    fig.suptitle("Dense AffNet upright shapes' components")

    idx = 0
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 1:
                continue
            img_t = K.tensor_to_image(aff_features[:, :, i, j])
            axs[idx].set_title("shapes[{}, {}]".format(i, j))
            axs[idx].imshow(img_t)
            idx = idx + 1
    plt.show(block=False)


def visualize_covered_pixels_and_connected_comp(conf, ts_phis, cover_idx, img_name, components_indices, valid_components_dict):

    show = conf.get(CartesianConfig.show_dense_affnet_components, False)
    if not show:
        return

    center_names = {-3: "sky",
                    -2: "identity eq. class",
                    -1: "no valid center"}
    l = 3 + len(ts_phis)
    columns = 4 if l > 3 else l
    rows = (l - 1) // 4 + 1
    fig, axs = plt.subplots(rows, columns, figsize=(10, 10))
    dense_affnet_filter = conf.get("affnet_dense_affnet_filter", None)
    use_orienter = conf.get(CartesianConfig.affnet_dense_affnet_use_orienter, "True")
    title = "{} - pixels of shapes covered by covering sets\ndense_affnet_filter={},use_orienter={} ".format(img_name, dense_affnet_filter, use_orienter)
    fig.suptitle(title)

    pxs = cover_idx.shape[0] * cover_idx.shape[1]

    for i in range(-3, len(ts_phis)):
        mask = cover_idx == i
        center_name = center_names.get(i, "covering set {}".format(i))

        idx = i + 3
        r = idx // 4
        c = idx % 4

        fraction = mask.sum() / pxs * 100
        axis = axs[r, c] if rows > 1 else axs[c]
        axis.set_title("{} pxs({:.02f}%)\n{}".format(mask.sum(), fraction, center_name))
        axis.imshow(mask)

    plt.show(block=False)

    get_and_show_components(components_indices,
                            valid_components_dict,
                            show=True,
                            save=False,
                            file_name=img_name)


def filter_components(component_idxs, fraction_threshold):

    component_size_threshold = component_idxs.shape[0] * component_idxs.shape[1] * fraction_threshold
    _, counts = torch.unique(component_idxs, return_counts=True)
    counts_mask = counts > component_size_threshold
    valid_component_dict = {}
    for i in range(3, len(counts_mask)):
        if counts_mask[i]:
            valid_component_dict[i - 3] = i - 3

    # NOTE compatibility with the existing code
    component_idxs = component_idxs.numpy()
    return component_idxs, valid_component_dict


def get_eligible_components(component_idxs, conf, valid_length):
    """
    :param component_idxs:
    :param conf:
    :param valid_length: I think effectively component_idxs.max() + 1
    :return:
    """

    fraction_threshold = conf.get(CartesianConfig.affnet_dense_affnet_cc_fraction_th, 0.008)
    enforce_cc = conf.get(CartesianConfig.affnet_dense_affnet_enforce_connected_components, True)
    if enforce_cc:
        components_indices, valid_components_dict = get_connected_components(component_idxs,
                                                                             range(valid_length),
                                                                             fraction_threshold=fraction_threshold)
    else:
        components_indices, valid_components_dict = filter_components(component_idxs, fraction_threshold)

    return components_indices, valid_components_dict


def apply_affnet_filter(gs_timg, conf):
    dense_affnet_filter = conf.get("affnet_dense_affnet_filter", None)
    if dense_affnet_filter is not None:
        print("WARNING: affnet_filter (={}) is being used".format(dense_affnet_filter), file=sys.stderr)
        gs_timg = gs_timg[:, :, ::dense_affnet_filter, ::dense_affnet_filter]
    return gs_timg, dense_affnet_filter


def add_affnet_coodrs_to_lafs(lafs):
    coords = affnet_coords(lafs.shape[:2])
    # (H, W) (shape -> mesh-grid) => (H, W) (geometric coords)
    lafs[:, :, 0, 2] = coords[1]
    lafs[:, :, 1, 2] = coords[0]


def possibly_apply_orienter(gs_timg, lafs, dense_affnet, conf):

    use_orienter = conf.get(CartesianConfig.affnet_dense_affnet_use_orienter, "True")
    if use_orienter:
        add_affnet_coodrs_to_lafs(lafs)
        orienter = KF.LAFOrienter(dense_affnet.patch_size, angle_detector=KF.OriNet(True))
        lafs_flat = lafs.reshape(1, lafs.shape[0] * lafs.shape[1], 2, 3)
        all_size = lafs_flat.shape[1]
        affnet_dense_affnet_batch = conf.get(CartesianConfig.affnet_dense_affnet_batch, None)
        if affnet_dense_affnet_batch is None:
            batch_size = all_size
        else:
            batch_size = affnet_dense_affnet_batch

        for i in range((all_size - 1) // batch_size + 1):
            l = i * batch_size
            u = min((i + 1) * batch_size, all_size)
            lafs_flat[:, l:u] = orienter(lafs_flat[:, l:u], gs_timg)
        lafs = lafs_flat.reshape(lafs.shape[0], lafs.shape[1], 2, 3)

    return lafs


def possibly_invert_lin_features(lin_features, conf):
    invert_first = conf.get("invert_first", True)
    # NOTE backward compatibility
    assert invert_first
    if invert_first:
        lin_features = torch.inverse(lin_features)
    return lin_features


def handle_upsample_early(upsample_early, cover_idx, dense_affnet_filter):
    if upsample_early:
        cover_idx = affnet_upsample(cover_idx)
        if dense_affnet_filter is not None:
            cover_idx = torch_upsample_factor(cover_idx, dense_affnet_filter)
    return cover_idx


def handle_upsample_late(upsample_early, components_indices, dense_affnet_filter):
    # NOTES
    # a) upsample_early is usually True, even though False may be more sensible
    # b) with False the data is transformed from numpy to torch and back
    # c) the most useful thing to do would be to redo the get_connected_components to work on torch
    # d) not so sure about the retyping (uint32 vs int32)
    if not upsample_early:
        assert np.all(components_indices < 256), "could not retype to np.uint8"
        t = torch.from_numpy(components_indices).to(torch.float)
        components_indices = affnet_upsample(t).to(torch.int32)
        if dense_affnet_filter is not None:
            components_indices = torch_upsample_factor(components_indices, dense_affnet_filter)
        components_indices = components_indices.numpy()
    return components_indices


def affnet_clustering(img, img_name, dense_affnet, conf, upsample_early, use_cuda=False):
    """
    The main function to call here.
    NOTE: especially when the orienter is on the lafs are very expensive to compute
    (so ideally some caching scheme would come in handy, BUT the "simple" step of producing dense lafs
    is still affected by some params)
    :param img:
    :param img_name:
    :param dense_affnet:
    :param conf:
    :param upsample_early:
    :param use_cuda:
    :return:
    """
    gs_timg = K.image_to_tensor(img, False).float() / 255.
    gs_timg = K.color.bgr_to_grayscale(gs_timg)
    gs_timg, dense_affnet_filter = apply_affnet_filter(gs_timg, conf)

    with torch.no_grad():
        lafs = dense_affnet(gs_timg)
        lafs = possibly_apply_orienter(gs_timg, lafs, dense_affnet, conf)
        show_affnet_features(lafs, conf)

        non_sky_mask = get_nonsky_mask_torch(img, lafs.shape[0], lafs.shape[1], use_cuda=use_cuda)
        non_sky_mask_flat = non_sky_mask.reshape(-1, 1)[:, 0]

        lin_features = lafs[:, :, :, :2]
        lin_features = possibly_invert_lin_features(lin_features, conf)

        _, _, ts1, phis1 = decompose_lin_maps_lambda_psi_t_phi(lin_features, asserts=False)
        covering: CoveringParams = CoveringParams.get_effective_covering_by_cfg(conf)
        ts_phis, cover_idx = winning_centers(covering, ts1.reshape(1, -1), phis1.reshape(1, -1), conf, return_cover_idxs=True, valid_px_mask=non_sky_mask_flat)

        # NOTE: index value convention
        # range(len(ts_phis)) -> all_valid
        # -3 sky
        # -2 identity equivalence class
        # -1 no valid center
        cover_idx = cover_idx.reshape(lin_features.shape[:2])
        cover_idx[~non_sky_mask] = -3

        cover_idx = handle_upsample_early(upsample_early, cover_idx, dense_affnet_filter)
        components_indices, valid_components_dict = get_eligible_components(cover_idx, conf, len(ts_phis))
        components_indices = handle_upsample_late(upsample_early, components_indices, dense_affnet_filter)

        visualize_covered_pixels_and_connected_comp(conf, ts_phis, cover_idx, img_name, components_indices, valid_components_dict)

        return ImageData(img=img,
                         real_K=None,
                         key_points=None,
                         descriptions=None,
                         normals=None,
                         ts_phis=ts_phis,
                         components_indices=components_indices,
                         valid_components_dict=valid_components_dict)


# DEMO SEGMENT
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


def demo_process_img(img_path, dense_affnet, hardnet_descriptor, use_cuda=False):

    img = read_img(img_path)
    img_name = img_path.split("/")[-1]

    conf = {"show_affnet": True,
            "affnet_covering_fraction_th": 0.97,
            "affnet_covering_max_iter": 100,
            "invert_first": True,
            "affnet_include_all_from_identity": True,
            "affnet_no_clustering": False,
            "affnet_covering_type": "dense_cover",
            }

    img_data = affnet_clustering(img, img_name, dense_affnet, conf, upsample_early=True, use_cuda=use_cuda)

    conf["affnet_covering_max_iter"] = 1
    affnet_rectify(img_path.split("/")[-1],
                   hardnet_descriptor,
                   img_data,
                   conf,
                   device=torch.device('cpu'),
                   params_key="",
                   stats_map={})


def affnet_clustering_demo():

    dense_affnet = DenseAffNet(True)
    hardnet_descriptor = get_default_hardnet()

    scene_info = SceneInfo.read_scene("scene1")
    file_names, _ = scene_info.get_megadepth_file_names_and_dir(None, None)
    for idx, depth_data_file_name in enumerate(file_names[:2]):
        demo_process_img(scene_info.get_img_file_path(depth_data_file_name[:-4]), dense_affnet, hardnet_descriptor, use_cuda=False)


def affnet_upsample_test():

    h = 473
    w = 263

    input = torch.arange(1, h * w + 1, dtype=torch.float32).view(h, w)
    out = affnet_upsample(input)
    print(out.shape)


if __name__ == "__main__":
    affnet_upsample_test()
    print("Done")

