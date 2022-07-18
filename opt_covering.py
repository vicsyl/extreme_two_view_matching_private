import matplotlib.pyplot as plt
import torch
import math
import logging
import pickle
import numpy as np

from config import CartesianConfig
from dataclasses import dataclass
from matplotlib.patches import Circle
from utils import Timer
from img_utils import create_plot_only_img
from connected_components import get_and_show_components
from utils import adjust_affine_transform, timer_label_decorator

@dataclass
class CoveringParams:

    r_max: float
    t_max: float
    ts_opt: list
    phis_opt: list
    name: str

    @staticmethod
    def log_1_8_covering():
        return CoveringParams(
            r_max=1.8,
            t_max=6.0,
            ts_opt=[2.88447, 6.2197],
            phis_opt=[math.pi / 8.0, math.pi / 16.0],
            name="log_1_8_covering")

    @staticmethod
    def log_1_8_covering_denser():
        return CoveringParams(
            r_max=1.8,
            t_max=6.0,
            # NOTE just pretty randomly populated
            ts_opt=[2.2, 2.88447, 4.28, 6.2197],
            phis_opt=[math.pi / 8.0, math.pi / 10.0, math.pi / 12.0, math.pi / 16.0],
            name="log_1_8_covering_denser")

    @staticmethod
    def log_1_8_covering_densest():
        return CoveringParams(
            r_max=1.8,
            t_max=6.0,
            # NOTE just pretty randomly populated
            ts_opt=[2.2, 2.5, 2.88447, 3.5, 4.28, 5.5, 6.2197],
            phis_opt=[math.pi / 16.0, math.pi / 20.0, math.pi / 24.0, math.pi / 28.0, math.pi / 32.0, math.pi / 36.0, math.pi / 40.0],
            name="log_1_8_covering_densest")

    @staticmethod
    def dense_covering_original():
        return CoveringParams(
            r_max=1.8,
            t_max=6.0,
            ts_opt=[2.2, 2.5],
            phis_opt=[math.pi / 32.0, math.pi / 32.0],
            name="narrow_covering")

    # CNN-ASSISTED COVERINGS IN THE SPACE OF TILTS:
    # BEST AFFINE INVARIANT PERFORMANCES WITH THE SPEED OF CNNS
    # (1.7, 5.8) - BUT DENSE!
    @staticmethod
    def dense_covering_1_7():
        bands = 6
        lg_sp = torch.logspace(math.log(2.0, 10), math.log(6.2197, 10), bands)
        return CoveringParams(
            r_max=1.7,
            t_max=5.8,
            ts_opt=lg_sp,
            phis_opt=[math.pi / 32.0] * bands,
            name="narrow_covering")

    # CNN-ASSISTED COVERINGS IN THE SPACE OF TILTS:
    # BEST AFFINE INVARIANT PERFORMANCES WITH THE SPEED OF CNNS
    # (1.7, 5.8) - BUT SPARSE!
    @staticmethod
    def sparse_covering_1_7():
        return CoveringParams(
            r_max=1.7,
            t_max=5.8,
            ts_opt=[2.88447, 6.2197],
            phis_opt=[math.pi / 16.0] * 2,
            name="narrow_covering")

    @staticmethod
    def sparse_covering_1_8_corrected():
        return CoveringParams(
            r_max=1.8,
            t_max=6.0,
            ts_opt=[2.88447, 6.2197],
            phis_opt=[math.pi / 8.0, math.pi / 16.0],
            name="sparse_covering_1_8_corrected")

    @staticmethod
    def get_effective_covering_by_cfg(config):
        covering_type = config["affnet_covering_type"]
        if covering_type == "mean":
            tilt_r_exp = config.get("affnet_tilt_r_ln", 1.7)
            max_tilt_r = config.get("affnet_max_tilt_r", 5.8)
            return CoveringParams(r_max=tilt_r_exp,
                                  t_max=5.8,
                                  ts_opt=None,
                                  phis_opt=None,
                                  name="mean like covering - r_max={}, t_max={}".format(tilt_r_exp, max_tilt_r))
        else:
            return CoveringParams.get_effective_covering(covering_type)

    @staticmethod
    def get_effective_covering(covering_type):
        if covering_type == "dense_cover_original":
            return CoveringParams.dense_covering_original()
        elif covering_type == "dense_cover":
            return CoveringParams.dense_covering_1_7()
        elif covering_type == "sparse_cover":
            return CoveringParams.sparse_covering_1_7()
        else:
            raise ValueError("Unknown covering type: {}".format(covering_type))

    def covering_coordinates(self):
        t_phi_list = []
        for index, t_opt in enumerate(self.ts_opt):
            for phi in torch.arange(start=0.0, end=math.pi, step=self.phis_opt[index]):
                t_phi_list.append((t_opt, phi))

        return torch.tensor(t_phi_list).T

    def covering_coordinates_count(self):
        # include the identity class
        count = 1
        for index in range(len(self.ts_opt)):
            count = count + len(torch.arange(start=0.0, end=math.pi, step=self.phis_opt[index]))
        return count


def distance_matrix(t1, t2, phi1, phi2):
    """
    t1, t2 tilts, not their logs!!
    """
    t1 = t1.unsqueeze(1).expand(-1, t2.shape[0])
    phi1 = phi1.unsqueeze(1).expand(-1, phi2.shape[0])
    t2 = t2.unsqueeze(0).expand(t1.shape[0], -1)
    phi2 = phi2.unsqueeze(0).expand(phi1.shape[0], -1)
    dist = (t1 / t2 + t2 / t1) * torch.cos(phi1 - phi2) ** 2 + (t1 * t2 + 1.0 / t2 * t1) * torch.sin(phi1 - phi2) ** 2
    dist = dist / 2
    return dist


def distance_matrix_concise(centers, data):
    """
    :param centers:
    :param data:
    :return:
    """
    t1, phi1 = centers[0], centers[1]
    t2, phi2 = data[0], data[1]
    return distance_matrix(t1, t2, phi1, phi2)


def draw_identity_data(ax, data, r):
    data_around_identity_mask = data[0] < r
    in_data = data[:, data_around_identity_mask]
    opt_conv_draw(ax, in_data, 'c', 0.5)


def vote(covering_params, data, fraction_th, iter_th, conf, return_cover_idxs=False):
    """
    Assumes the data is prefiltered with e.g. sky mask. Data with t > t_max are
        a) marked with cover_idx == -4 if not covered (difference between all with t > t_max and those uncovered are
           logged on debug level)
        b) ignored wrt. fraction_th

    NOTE - the cover_idx membership array has the size of data

    :param covering_params:
    :param data:
    :param fraction_th:
    :param iter_th:
    :param return_cover_idxs:
    :param conf:
    :return: winning_centers , cover_idx (=None return_cover_idxs == False)
        winning_centers: rows of with 2 columns - (tau_i, phi_i)
        cover_idx: index of centers for the data points
                    -1: no winning center
                    -2: identity equivalence class
                   (-3: sky - not here)
                    -4: uncovered off
                  i>=0: winning center index i
    """

    # NOTE: filtered_data works as index-less data points, whereas cover_idx are filters across all indices
    # if everything is done across all indices, it may get simpler

    centers = covering_params.covering_coordinates()
    r_param = covering_params.r_max
    t_max = covering_params.t_max

    Timer.start_check_point("vote_covering_centers")
    closest_winning_center = conf.get(CartesianConfig.sof_coverings_closest_winning_center, True)

    r_ball_distance_new = (r_param ** 2 + 1) / (2 * r_param)

    # TODO delete the following 4 lines of code
    r_old = math.log(r_param)
    r_ball_distance_old = (math.exp(2 * r_old) + 1) / (2 * math.exp(r_old))
    # CONTINUE here + run & think about stats and visos
    assert math.fabs(r_ball_distance_old - r_ball_distance_new) < 1.0e-10
    assert math.exp(r_old) == r_param

    # TODO effective_data_size - should be data and valid_px_mask and ~data_completely_off (but irrespective of data_around_identity_mask)
    data_completely_off = data[0] > t_max
    print("all data points: {}".format(data.shape[1]))
    init_data_size = data.shape[1] - data_completely_off.sum()
    print("data points completely off: {}".format(data_completely_off.sum()))

    data_around_identity_mask = data[0] < r_param
    # NOTE data_around_identity_mask and data_completely_off are disjunctive sets
    filtered_data = data[:, ~data_around_identity_mask & ~data_completely_off]

    cover_idx = None
    if return_cover_idxs and not closest_winning_center:
        cover_idx = torch.ones(data.shape[1]) * -1
        cover_idx[data_around_identity_mask] = -2
        cover_idx[data_completely_off] = -4
        logging.debug("initial data points with t > t_max: {}".format(data_completely_off.sum()))

    iter_finished = 0
    winning_centers = []
    rect_fraction = 1 - filtered_data.shape[1] / init_data_size
    while rect_fraction < fraction_th and iter_finished < iter_th:

        distances = distance_matrix_concise(centers, filtered_data)
        votes = (distances < r_ball_distance_new)
        votes_count = votes.sum(axis=1)
        sorted, indices = torch.sort(votes_count, descending=True)
        data_in_mask = votes[indices[0]]

        if return_cover_idxs and not closest_winning_center:
            distances_all = distance_matrix_concise(centers[:, indices[0]:indices[0] + 1], data)
            votes_all = (distances_all < r_ball_distance_new)
            # & on bools?
            votes_new = votes_all[0] & (cover_idx == -1)
            cover_idx[votes_new] = iter_finished

        filtered_data = filtered_data[:, ~data_in_mask]
        rect_fraction = 1 - filtered_data.shape[1] / init_data_size

        winning_center = centers[:, indices[0]]
        winning_centers.append((winning_center[0].item(), winning_center[1].item()))
        iter_finished += 1

    winning_centers = torch.tensor(winning_centers)

    Timer.end_check_point("vote_covering_centers")

    if return_cover_idxs:
        if closest_winning_center:
            cover_idx = torch.ones(data.shape[1]) * -1
            cover_idx[data_completely_off] = -4
            logging.debug("initial data points with t > t_max: {}".format(data_completely_off.sum()))

            distance_for_identity = conf.get(CartesianConfig.sof_coverings_distance_for_identity, False)

            if distance_for_identity:
                winning_centers_dist = torch.hstack((winning_centers.t(), torch.tensor([[1.0], [0.0]])))
            else:
                winning_centers_dist = winning_centers.t()

            distances = distance_matrix_concise(winning_centers_dist, data)

            minimal_wc_distances_values_indices = torch.min(distances, 0)
            mask = minimal_wc_distances_values_indices.values < r_ball_distance_new
            cover_idx[mask] = minimal_wc_distances_values_indices.indices[mask].type(cover_idx.dtype)

            if distance_for_identity:
                cover_idx[cover_idx == winning_centers_dist.shape[1] - 1] = -2
            else:
                cover_idx[data_around_identity_mask] = -2

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("not covered data points with t > t_max: {}".format((cover_idx[data_completely_off] == -4).sum()))
    return winning_centers, cover_idx


def opt_conv_draw_ellipses(ax, cov_params, centers, thickness=0.005, color="b"):

    log_max_radius = math.log(cov_params.t_max)
    # log_unit_radius_bigger = math.log(cov_params.r_max * 1.05)
    # log_unit_radius = math.log(cov_params.r_max)
    # rhs = (math.exp(2 * log_unit_radius) + 1) / (2 * math.exp(log_unit_radius))
    rhs = (cov_params.r_max ** 2 + 1) / (2 * cov_params.r_max)

    factor = 1.4
    extend = factor * log_max_radius
    range_x = torch.arange(start=-extend, end=extend, step=thickness)
    range_y = torch.arange(start=0, end=extend, step=thickness)
    grid_x, grid_y = torch.meshgrid(range_x, range_y)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    ts = torch.exp(torch.sqrt(grid_x ** 2 + grid_y ** 2))
    phis = torch.atan(grid_x / grid_y)

    distances_close = torch.abs(distance_matrix(ts, centers[0], phis, centers[1]) - rhs)
    distances_close = distances_close.min(axis=1)[0]

    grid_x = grid_x[distances_close < thickness]
    grid_y = grid_y[distances_close < thickness]

    ax.plot(grid_x, grid_y, 'o', color=color, markersize=0.5)


def opt_conv_draw(ax, ts_phis, color, size, shape='o'):

    tilts_logs = torch.log(ts_phis[0])
    xs = torch.cos(ts_phis[1]) * tilts_logs
    ys = torch.sin(ts_phis[1]) * tilts_logs
    ax.plot(xs, ys, shape, color=color, markersize=size)


# NOTE probably can be inline
def set_cov_axis(cov_params: CoveringParams, ax, positive_only=False):
    log_max_radius = math.log(cov_params.t_max)
    factor = 1.4
    abs_lim = factor * log_max_radius
    ax.set_xlim((-abs_lim, abs_lim))
    plt.yticks(range(3), ("0", "1", "2"), size="large")
    plt.xticks(range(-2, 3), ("2", "1", "0", "1", "2"), size="large")

    plt.xlabel("log(τ).cos(φ)", fontsize='x-large')
    plt.ylabel("log(τ).sin(φ)", fontsize='x-large')

    if positive_only:
        ax.set_ylim((0, abs_lim))
    else:
        ax.set_ylim((-abs_lim, abs_lim))


# TODO centralize with opt_cov_prepare_plot_custom
def opt_cov_prepare_plot_custom(ax, cov_params: CoveringParams, title=None, positive_only=False):

    if title is not None:
        plt.title(title)

    log_max_radius = math.log(cov_params.t_max)
    log_unit_radius = math.log(cov_params.r_max)

    ax.set_aspect(1.0)
    set_cov_axis(cov_params, ax, positive_only)

    circle = Circle((0, 0), log_max_radius, color='r', fill=False, lw=2.0)
    ax.add_artist(circle)
    circle = Circle((0, 0), log_unit_radius, color='r', fill=False, lw=2.0)
    ax.add_artist(circle)

    return ax


def opt_cov_prepare_plot(cov_params: CoveringParams, title=None):

    fig, ax = plt.subplots(figsize=(10, 10))
    if title is not None:
        plt.title(title)

    log_max_radius = math.log(cov_params.t_max)
    log_unit_radius = math.log(cov_params.r_max)

    ax.set_aspect(1.0)
    set_cov_axis(cov_params, ax)

    circle = Circle((0, 0), log_max_radius, color='r', fill=False)
    ax.add_artist(circle)
    circle = Circle((0, 0), log_unit_radius, color='r', fill=False)
    ax.add_artist(circle)

    return ax


def draw_covered_data(ax, center, data, r_max, color):
    r_log = math.log(r_max)
    rhs = (math.exp(2 * r_log) + 1) / (2 * math.exp(r_log))
    distances = distance_matrix(center[0, None], data[0], center[1, None], data[1])
    votes = (distances[0] < rhs)
    data_in = data[:, votes]
    opt_conv_draw(ax, data_in, color, 0.5)


@timer_label_decorator()
def visualize_covered_pixels_and_connected_comp(conf, ts_phis, cover_idx, img_name, components_indices_arg, valid_components_dict_arg):

    show = conf.get(CartesianConfig.show_dense_affnet_components, False)
    if not show:
        return

    valid_components_dict = valid_components_dict_arg.copy()
    components_indices = np.copy(components_indices_arg)

    # identity
    valid_components_dict[-2] = -2

    # sky
    valid_components_dict[-1] = -1

    # TODO hack -> sky => no valid components
    components_indices[components_indices == -3] = -1

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

    # TODO - handle save and path properly
    # switch the save flag if necessary
    get_and_show_components(components_indices,
                            valid_components_dict,
                            show=True,
                            save=False,
                            path="./work/",
                            file_name=img_name)

    # TODO clean this up

    not_temporarily_closed = False
    if not_temporarily_closed:
        colors = [
            [255, 0, 0],
            [255, 255, 0],
            [1, 1, 1],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [0, 255, 255],
            [128, 0, 0],
            [0, 128, 0],
            [0, 0, 128],
        ]
        color_ix = 0

        color_map_ix = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

        As = np.array([
            [[1.0, 0.0, 0.0],
             [0.0, 0.5, 0.0],
             [0.0, 0.0, 1.0]
             ],
            [[1.0, 0.8, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             ],
            [[1.0, -0.5, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             ],
            [[0.71, -0.71, 0.0],
             [0.71, 0.71, 0.0],
             [0.0, 0.0, 1.0],
             ],
        ])

        As_map_ix = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3}

        for i in range(-3, len(ts_phis)):
            img_to_show = np.zeros(((*cover_idx.shape[:2], 3)))
            img_to_show[cover_idx == i] = colors[color_map_ix[color_ix] % len(colors)]

            create_plot_only_img(title, img_to_show, h_size_inches=6, transparent=True)
            plt.savefig("./work/segments_dense_affnet_{}".format(color_ix), dpi=24, transparent=True, facecolor=(0.0, 0.0, 0.0, 0.0))

            A, bb = adjust_affine_transform(img_to_show, None, As[As_map_ix[color_ix]])
            img_to_show = np.int32(cv.warpPerspective(np.float32(img_to_show), A, bb))

            create_plot_only_img(title, img_to_show, h_size_inches=6, transparent=True)
            plt.savefig("./work/segments_dense_affnet_transformed_{}".format(color_ix), dpi=24, transparent=True, facecolor=(0.0, 0.0, 0.0, 0.0))

            color_ix = color_ix + 1


# TODO I am afraid the transparency is not handled (i.e. it's enabled somewhere else)
def prepare_coverings_plot(covering_params, data, winning_centers, cover_ix, with_title, with_axis, draw_sky_over=False):

    # TODO handle the colors
    # b : blue.
    # g : green.
    # r : red.
    # c : cyan.
    # m : magenta.
    # y : yellow.
    # k : black.
    # w : white.

    colors_names = [
        ("r", "red"),
        ("g", "green"),
        ("b", "blue"),
        ("y", "yellow"),
        ("c", "cyan"),
        ("m", "magenta")]

    indices_names = {-4: "off not covered", -3: "sky", -2: "identity", -1: "not covered"}

    title = None
    if with_title:
        colors_legend = []
        for i in range(-4, len(winning_centers)):
            # e.g. sky may be skipped
            if (cover_ix == i).sum() == 0:
                continue
            indices_name = indices_names.get(i, i)
            if i % 4 == 0:
                indices_name = "\n{}".format(indices_name)
            color_name = colors_names[i % len(colors_names)][1] if i != -3 else "black"
            colors_legend.append("{}={}".format(indices_name, color_name))
        title = "Covering the space of tilts:{}".format(", ".join(colors_legend))

    ax = opt_cov_prepare_plot(covering_params, title)
    if not with_axis:
        ax.set_axis_off()

    opt_conv_draw(ax, data, "k", 1.0)

    for i in range(-4, len(winning_centers)):

        color_letter = colors_names[i % len(colors_names)][0]
        data_to_draw = data[:, cover_ix == i]
        opt_conv_draw(ax, data_to_draw, color_letter, size=0.5)

        if i >= 0:
            wc = winning_centers[i]
            opt_conv_draw(ax, wc, "k", 8.0, shape="x")

    if draw_sky_over:
        data_to_draw = data[:, cover_ix == -3]
        opt_conv_draw(ax, data_to_draw, "k", size=0.5)


# TODO I am afraid the transparency is not handled (i.e. it's enabled somewhere else)
def prepare_coverings_plot_closest(covering_params, data, winning_centers, with_title, with_axis):

    # TODO handle the colors
    colors = ["r", "g", "b", "y"]
    colors_unrolled = [colors[i % len(colors)] for i in range(len(winning_centers) - 1, -1, -1)]

    if with_title:
        title = "Covering the space of tilts:\n not-covered - black, identity eq. class - cyan".format(", ".join(colors_unrolled))
    else:
        title = None

    ax = opt_cov_prepare_plot(covering_params, title)
    if not with_axis:
        ax.set_axis_off()

    opt_conv_draw(ax, data, "k", 1.0)

    for i in range(len(winning_centers) - 1, -1, -1):
        wc = winning_centers[i]
        color = colors_unrolled[i]
        draw_covered_data(ax, wc, data, covering_params.r_max, color)
        opt_conv_draw(ax, wc, "k", 8.0, shape="x")

    draw_identity_data(ax, data, covering_params.r_max)


@timer_label_decorator()
def potentially_show_sof(covering_params, data, winning_centers, config, cover_idx=None, enforce_show=False):

    show_affnet = config.get(CartesianConfig.show_affnet, enforce_show)
    if show_affnet:
        if cover_idx is None:
            prepare_coverings_plot_closest(covering_params, data, winning_centers, with_title=True, with_axis=True)
        else:
            prepare_coverings_plot(covering_params, data, winning_centers, cover_idx, with_title=True, with_axis=True)
        plt.show(block=False)

    save_affnet_coverings = config.get(CartesianConfig.save_affnet_coverings, False)
    if save_affnet_coverings:
        if cover_idx is None:
            prepare_coverings_plot_closest(covering_params, data, winning_centers, with_title=False, with_axis=False)
        else:
            prepare_coverings_plot(covering_params, data, winning_centers, cover_idx, with_title=False, with_axis=False)
        # TODO externalize the path
        plt.savefig("./work/covering_dense", dpi=48)
        plt.close()


def vote_test():

    logging.getLogger().setLevel(logging.DEBUG)

    covering_params = CoveringParams.dense_covering_1_7()

    with open("resources/covering_data.pkl", "rb") as f:
        sot_data = pickle.load(f)

    fraction_th = 0.95
    iter_th = 100
    show_affnet_config = {CartesianConfig.show_affnet: True}

    # closest to the winning center
    ret_winning_centers1, cv1 = vote(covering_params,
                                    sot_data,
                                    fraction_th,
                                    iter_th,
                                    return_cover_idxs=True, conf={})

    potentially_show_sof(covering_params, sot_data, ret_winning_centers1, show_affnet_config, cover_idx=cv1)

    # closest to the winning center, distance_for_identity = True
    ret_winning_centers_id, cv_id = vote(covering_params,
                                         sot_data,
                                         fraction_th,
                                         iter_th,
                                         return_cover_idxs=True, conf={CartesianConfig.sof_coverings_distance_for_identity: True})

    potentially_show_sof(covering_params, sot_data, ret_winning_centers_id, show_affnet_config, cover_idx=cv_id)

    # old method, new implementation
    ret_winning_centers2, cv2 = vote(covering_params,
                                    sot_data,
                                    fraction_th,
                                    iter_th,
                                    return_cover_idxs=True, conf={CartesianConfig.sof_coverings_closest_winning_center: False})

    potentially_show_sof(covering_params, sot_data, ret_winning_centers2, show_affnet_config, cover_idx=cv2)

    # old method, old implementation
    covering_coords = covering_params.covering_coordinates()
    ret_winning_centers3, cv3 = vote_old(covering_coords, sot_data, covering_params.r_max,
                                        fraction_th,
                                        iter_th,
                                        return_cover_idxs=True,
                                        t_max=covering_params.t_max)

    potentially_show_sof(covering_params, sot_data, ret_winning_centers3, config={CartesianConfig.show_affnet: True}, cover_idx=cv3)


def demo():

    #covering_params = CoveringParams.log_1_8_covering()
    covering_params = CoveringParams.sparse_covering_1_8_corrected()
    print("count: {}".format(covering_params.covering_coordinates_count()))

    data_count = 5000
    data = torch.rand(2, data_count)
    data[0] = torch.abs(data[0] * 5.0 + 1)
    data[1] = data[1] * math.pi

    #opt_conv_draw(ax, data, "b", 1.0)

    covering_centers = covering_params.covering_coordinates()

    fig, ax = plt.subplots(figsize=(10, 10))

    opt_cov_prepare_plot_custom(ax, covering_params, title=None, positive_only=True) # "Covering - centers, cover sets, data points...")

    opt_conv_draw(ax, covering_centers, "k", 8.0)
    ax.plot(0, 0, 'o', color="k", markersize=8.0)

    opt_conv_draw_ellipses(ax, covering_params, covering_centers, thickness=0.005, color="blue")

    plt.savefig("work/sot1.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

    winning_centers = vote(covering_params, data, fraction_th=0.6, iter_th=30, conf={})

    # for i, wc in enumerate(winning_centers):
    #     draw_in_center(ax, wc, data, covering_params.r_max)
    #     opt_conv_draw(ax, wc, "b", 5.0)

    #draw_identity_data(ax, data, covering_params.r_max)


def vote_old(centers, data, r_param, fraction_th, iter_th, return_cover_idxs=False, valid_px_mask=None, t_max=None):
    """
    :param centers:
    :param data:
    :param r_param:
    :param fraction_th:
    :param iter_th:
    :param return_cover_idxs:
    :param valid_px_mask:
    :return: winning_centers (, cover_idx - if return_cover_idxs is True)
        winning_centers: rows of with 2 columns - (tau_i, phi_i)
        cover_idx: index of centers for the data points
                    -1 : no winning center
                    -2 : identity equivalence class
                    >=0: winning center index
    """

    # NOTE: filtered_data works as index-less data points, whereas cover_idx are filters across all indices
    # if everything is done across all indices, it may get simpler

    Timer.start_check_point("vote_covering_centers")

    if valid_px_mask is None:
        valid_px_mask = torch.ones(data.shape[1], dtype=torch.bool)

    r_ball_distance_new = (r_param ** 2 + 1) / (2 * r_param)

    # TODO delete the following 4 lines of code
    r_old = math.log(r_param)
    r_ball_distance_old = (math.exp(2 * r_old) + 1) / (2 * math.exp(r_old))
    # CONTINUE here + run & think about stats and visos
    assert math.fabs(r_ball_distance_old - r_ball_distance_new) < 1.0e-10
    assert math.exp(r_old) == r_param

    data_around_identity_mask = data[0] < r_param

    init_filter = ~data_around_identity_mask & valid_px_mask

    # TODO effective_data_size - should be data and valid_px_mask and ~data_completely_off (but irrespective of data_around_identity_mask)
    if t_max is not None:
        data_completely_off = data[0] > t_max
        init_filter = init_filter & ~data_completely_off

    filtered_data = data[:, init_filter]

    if return_cover_idxs:
        cover_idx = torch.ones(data.shape[1]) * -1
        valid_identity_filter = data_around_identity_mask & valid_px_mask
        cover_idx[valid_identity_filter] = -2

    iter_finished = 0
    winning_centers = []
    rect_fraction = 1 - filtered_data.shape[1] / data.shape[1]
    while rect_fraction < fraction_th and iter_finished < iter_th:

        distances = distance_matrix(centers[0], filtered_data[0], centers[1], filtered_data[1])
        votes = (distances < r_ball_distance_new)
        votes_count = votes.sum(axis=1)
        sorted, indices = torch.sort(votes_count, descending=True)

        data_in_mask = votes[indices[0]]
        if return_cover_idxs:
            distances_all = distance_matrix_concise(centers[:, indices[0]:indices[0] + 1], data)
            votes_all = (distances_all < r_ball_distance_new)
            # & on bools?
            votes_new = votes_all[0] & (cover_idx == -1)
            cover_idx[votes_new] = iter_finished

        filtered_data = filtered_data[:, ~data_in_mask]
        rect_fraction = 1 - filtered_data.shape[1] / data.shape[1]

        winning_center = centers[:, indices[0]]
        winning_centers.append((winning_center[0].item(), winning_center[1].item()))
        iter_finished += 1

    Timer.end_check_point("vote_covering_centers")

    if return_cover_idxs:
        return torch.tensor(winning_centers), cover_idx
    else:
        return torch.tensor(winning_centers)


if __name__ == "__main__":
    demo()
    # vote_test()
