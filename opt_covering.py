import matplotlib.pyplot as plt
import torch
import math
from dataclasses import dataclass
from matplotlib.patches import Circle
from utils import Timer

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
        bands = 6
        return CoveringParams(
            r_max=1.7,
            t_max=5.8,
            ts_opt=[2.88447, 6.2197],
            phis_opt=[math.pi / 16.0] * 2,
            name="narrow_covering")

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
        count = 0
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
    opt_conv_draw(ax, in_data, 'c', 2)


def vote(centers, data, r_param, fraction_th, iter_th, return_cover_idxs=False, valid_px_mask=None, t_max=None):
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


def opt_conv_draw_ellipses(ax, cov_params, centers):

    log_max_radius = math.log(cov_params.t_max)
    log_unit_radius = math.log(cov_params.r_max)
    rhs = (math.exp(2 * log_unit_radius) + 1) / (2 * math.exp(log_unit_radius))

    factor = 1.2
    extend = factor * log_max_radius
    range_x = torch.arange(start=-extend, end=extend, step=0.005)
    range_y = torch.arange(start=-extend, end=extend, step=0.005)
    grid_x, grid_y = torch.meshgrid(range_x, range_y)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    ts = torch.exp(torch.sqrt(grid_x ** 2 + grid_y ** 2))
    phis = torch.atan(grid_x / grid_y)

    distances_close = torch.abs(distance_matrix(ts, centers[0], phis, centers[1]) - rhs)
    distances_close = distances_close.min(axis=1)[0]

    grid_x = grid_x[distances_close < 0.005]
    grid_y = grid_y[distances_close < 0.005]

    ax.plot(grid_x, grid_y, 'o', color="b", markersize=0.2)


def opt_conv_draw(ax, ts_phis, color, size, shape='o'):

    tilts_logs = torch.log(ts_phis[0])
    xs = torch.cos(ts_phis[1]) * tilts_logs
    ys = torch.sin(ts_phis[1]) * tilts_logs
    ax.plot(xs, ys, shape, color=color, markersize=size)


def opt_cov_prepare_plot(cov_params: CoveringParams, title):

    fig, ax = plt.subplots()
    plt.title(title)

    log_max_radius = math.log(cov_params.t_max)
    log_unit_radius = math.log(cov_params.r_max)

    ax.set_aspect(1.0)

    factor = 1.2
    ax.set_xlim((-factor * log_max_radius, factor * log_max_radius))
    ax.set_ylim((-factor * log_max_radius, factor * log_max_radius))

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
    opt_conv_draw(ax, data_in, color, 1)


def demo():

    #covering_params = CoveringParams.log_1_8_covering()
    covering_params = CoveringParams.dense_covering_1_7()
    print("count: {}".format(covering_params.covering_coordinates_count()))

    data_count = 5000
    data = torch.rand(2, data_count)
    data[0] = torch.abs(data[0] * 5.0 + 1)
    data[1] = data[1] * math.pi

    ax = opt_cov_prepare_plot(covering_params, title="Covering - centers, cover sets, data points...")

    #opt_conv_draw(ax, data, "b", 1.0)

    covering_centers = covering_params.covering_coordinates()
    opt_conv_draw(ax, covering_centers, "r", 5.0)

    #opt_conv_draw_ellipses(ax, covering_params, covering_centers)

    winning_centers = vote(covering_centers, data, covering_params.r_max, fraction_th=0.6, iter_th=30, t_max=covering_params.t_max)

    # for i, wc in enumerate(winning_centers):
    #     draw_in_center(ax, wc, data, covering_params.r_max)
    #     opt_conv_draw(ax, wc, "b", 5.0)

    #draw_identity_data(ax, data, covering_params.r_max)

    plt.show()


if __name__ == "__main__":
    demo()