#import sys
from datetime import datetime

from hard_net_descriptor import HardNetDescriptor
from normals_rotations import *
from rootsift_descriptor import RootSIFT

sys.path.append("./superpoint_forked")

from superpoint import SuperPointDescriptor

from config import *
from connected_components import get_connected_components, get_and_show_components
from depth_to_normals import *
from matching import match_epipolar, match_find_F_degensac, match_images_with_dominant_planes
from rectification import possibly_upsample_normals, get_rectified_keypoints
from img_utils import get_degrees_between_normals
from evaluation import *
from sky_filter import get_nonsky_mask
from clustering import Clustering, bilateral_filter

from affnet import affnet_rectify
from dense_affnet import DenseAffNet
from affnet_clustering import affnet_clustering

import matplotlib.pyplot as plt

two_hundred_permutation = [164, 90, 8, 35, 50, 112, 30, 51, 120, 78, 130, 134, 171, 5, 101, 147, 192, 72, 47, 156, 105,
                           22, 181, 129, 16, 198, 82, 100, 188, 159, 107, 86, 93, 151, 136, 96, 97, 83, 143, 0, 165,
                           185, 91, 7, 61, 12, 160, 92, 41, 184, 148, 76, 162, 157, 109, 20, 183, 17, 161, 132, 117,
                           178, 32, 111, 80, 153, 4, 180, 42, 116, 68, 95, 1, 189, 46, 170, 121, 139, 63, 58, 89, 177,
                           125, 75, 23, 167, 146, 2, 64, 94, 166, 145, 141, 6, 194, 197, 62, 172, 124, 193, 48, 24, 196,
                           85, 81, 60, 57, 88, 182, 126, 37, 169, 128, 39, 175, 11, 55, 40, 19, 65, 118, 84, 67, 69, 25,
                           43, 34, 168, 140, 137, 187, 150, 49, 186, 149, 59, 122, 144, 190, 9, 98, 174, 138, 102, 79,
                           66, 10, 110, 28, 29, 114, 77, 52, 123, 113, 108, 87, 33, 53, 199, 45, 179, 99, 135, 15, 73,
                           104, 131, 71, 31, 133, 176, 119, 191, 38, 155, 44, 3, 26, 18, 36, 154, 13, 173, 21, 27, 70,
                           152, 127, 54, 14, 163, 115, 103, 142, 56, 195, 158, 106, 74]


def parse_list(list_str: str):
    fields = list_str.split(",")
    fields = filter(lambda x: x != "", map(lambda x: x.strip(), fields))
    fields = list(fields)
    return fields


def possibly_expand_normals(normals):
    if len(normals.shape) == 1:
        normals = np.expand_dims(normals, axis=0)
    return normals


PROCESSING_IMG_FROM_SCRATCH_TAG = "processing_img_from_scratch"
COMPLETE_IMAGE_PAIR_MATCHING_TAG = "complete_image_pair_matching"


@dataclass
class Pipeline:

    # NOTE to be removed - just fot visualizations
    # counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stages = ["before_rectification", "before_matching", "final"]
    stages_map = {v: i for i, v in enumerate(stages)}

    def get_stage_number(self):
        return self.stages_map[self.config["pipeline_final_step"]]

    method = None
    file_name_suffix = None
    output_dir = None
    output_dir_prefix = None

    ransac_th = 0.5
    ransac_conf = 0.9999
    ransac_iters = 100000

    config = None
    cache_map = None
    # obsolete
    stats_map = {}
    # simple
    stats = {}

    chosen_depth_files = None

    show_input_img = False

    show_clusters = True
    save_clusters = True
    show_clustered_components = True
    save_clustered_components = True
    show_rectification = True
    save_rectification = True
    show_sky_mask = True
    save_sky_mask = True
    show_matching = True
    save_matching = True

    #matching
    feature_descriptor = None
    #matching_dir = None
    matching_difficulties = None
    matching_limit = None
    matching_pairs = None

    planes_based_matching = False
    use_degensac = False
    estimate_k = False
    focal_point_mean_factor = 0.5

    clip_angle = None

    knn_ratio_threshold = 0.85

    use_cached_img_data = True

    upsample_early = True

    # connected components
    connected_components_connectivity = 4
    connected_components_closing_size = None
    connected_components_flood_fill = False

    def setup_descriptor(self):
        self.feature_descriptor = Pipeline.setup_descriptor_static(self.config, self.device)

    @staticmethod
    def setup_descriptor_static(config, device=torch.device("cpu")):
        feature_descriptor = config["feature_descriptor"]
        n_features = config["n_features"]
        sift_octave_layers = config["sift_octave_layers"]
        sift_contrast_threshold = config["sift_contrast_threshold"]
        sift_edge_threshold = config["sift_edge_threshold"]
        sift_sigma = config["sift_sigma"]

        if feature_descriptor == "SIFT":
            feature_descriptor = cv.SIFT_create(n_features, sift_octave_layers, sift_contrast_threshold, sift_edge_threshold, sift_sigma)
        elif feature_descriptor == "BRISK":
            feature_descriptor = cv.BRISK_create(n_features)
        elif feature_descriptor == "SUPERPOINT":
            feature_descriptor = SuperPointDescriptor(path="./superpoint_forked/superpoint_v1.pth", device=device)
        elif feature_descriptor == "ROOT_SIFT":
            feature_descriptor = cv.SIFT_create(n_features, sift_octave_layers, sift_contrast_threshold, sift_edge_threshold, sift_sigma)
            feature_descriptor = RootSIFT(feature_descriptor)
        elif feature_descriptor == "HARD_NET":
            affnet_hard_net_filter = config.get("affnet_hard_net_filter", None)
            affnet_compute_laffs = config.get("affnet_compute_laffs", False)
            sift_detector = cv.SIFT_create(n_features, sift_octave_layers, sift_contrast_threshold, sift_edge_threshold, sift_sigma)
            feature_descriptor = HardNetDescriptor(sift_detector, affnet_compute_laffs, filter=affnet_hard_net_filter, device=device)
        return feature_descriptor

    @staticmethod
    def configure(config_file_name: str, args):

        # https://docs.python.org/2/library/configparser.html
        #     import configparser
        #     config = configparser.ConfigParser()
        #     config.read("config.txt")
        #     cf = config['settings']
        #     sc = cf['save_clusters']
        #     print()

        pipeline = Pipeline()
        config = CartesianConfig.get_default_cfg()

        with open(config_file_name) as f:
            for line in f:

                if line.strip().startswith("#"):
                    continue
                elif line.strip() == "":
                    continue

                k, v = line.partition("=")[::2]
                k = k.strip()
                v = v.strip()

                if k == "device":
                    if v == "cpu":
                        pipeline.device = torch.device("cpu")
                    elif v == "cuda":
                        pipeline.device = torch.device("cuda")
                    else:
                        raise Exception("Unknown param value for 'device': {}".format(v))
                elif k == "method":
                    pipeline.method = v
                elif k == "file_name_suffix":
                    pipeline.file_name_suffix = v
                elif k == "use_degensac":
                    pipeline.use_degensac = v.lower() == "true"
                elif k == "estimate_k":
                    pipeline.estimate_k = v.lower() == "true"
                elif k == "focal_point_mean_factor":
                    pipeline.focal_point_mean_factor = float(v)
                elif k == "knn_ratio_threshold":
                    pipeline.knn_ratio_threshold = float(v)
                elif k == "matching_difficulties_min":
                    matching_difficulties_min = int(v)
                elif k == "matching_difficulties_max":
                    matching_difficulties_max = int(v)
                elif k == "matching_limit":
                    pipeline.matching_limit = int(v)
                elif k == "planes_based_matching":
                    pipeline.planes_based_matching = v.lower() == "true"
                elif k == "output_dir":
                    pipeline.output_dir = v
                elif k == "show_input_img":
                    pipeline.show_input_img = v.lower() == "true"
                elif k == "show_matching":
                    pipeline.show_matching = v.lower() == "true"
                elif k == "save_matching":
                    pipeline.save_matching = v.lower() == "true"
                elif k == "show_clusters":
                    pipeline.show_clusters = v.lower() == "true"
                elif k == "save_clusters":
                    pipeline.save_clusters = v.lower() == "true"
                elif k == "show_clustered_components":
                    pipeline.show_clustered_components = v.lower() == "true"
                elif k == "save_clustered_components":
                    pipeline.save_clustered_components = v.lower() == "true"
                elif k == "show_rectification":
                    pipeline.show_rectification = v.lower() == "true"
                elif k == "save_rectification":
                    pipeline.save_rectification = v.lower() == "true"
                elif k == "show_sky_mask":
                    pipeline.show_sky_mask = v.lower() == "true"
                elif k == "save_sky_mask":
                    pipeline.save_sky_mask = v.lower() == "true"
                elif k == "do_flann":
                    Config.config_map[Config.key_do_flann] = v.lower() == "true"
                elif k == "matching_pairs":
                    pipeline.matching_pairs = parse_list(v)
                elif k == "chosen_depth_files":
                    pipeline.chosen_depth_files = parse_list(v)
                elif k == "use_cached_img_data":
                    pipeline.use_cached_img_data = v.lower() == "true"
                elif k == "output_dir_prefix":
                    pipeline.output_dir_prefix = v
                elif k == "ransac_th":
                    pipeline.ransac_th = float(v)
                elif k == "ransac_conf":
                    pipeline.ransac_conf = float(v)
                elif k == "ransac_iters":
                    pipeline.ransac_iters = int(v)
                elif k == "upsample_early":
                    pipeline.upsample_early = v.lower() == "true"
                elif k == "clip_angle":
                    if v.lower() == "none":
                        pipeline.clip_angle = None
                    else:
                        pipeline.clip_angle = int(v)
                elif k == "connected_components_connectivity":
                    value = int(v)
                    assert value == 4 or value == 8, "connected_components_connectivity must be 4 or 8"
                    pipeline.connected_components_connectivity = value
                elif k == "connected_components_closing_size":
                    if v.lower() == "none":
                        pipeline.connected_components_closing_size = None
                    else:
                        pipeline.connected_components_closing_size = int(v)
                elif k == "connected_components_flood_fill":
                    pipeline.connected_components_flood_fill = v.lower() == "true"
                else:
                    CartesianConfig.config_parse_line(k, v, config)

        pipeline.matching_difficulties = list(range(matching_difficulties_min, matching_difficulties_max))

        if args is not None and args.__contains__("output_dir"):
            pipeline.output_dir = args.output_dir
        elif pipeline.output_dir is None:
            pipeline.output_dir = append_all(pipeline, pipeline.output_dir_prefix)

        # NOTE planes_based_matching not really used anymore
        #assert not pipeline.planes_based_matching or pipeline.rectify, "rectification must be on for planes_based_matching"

        return pipeline, config

    def start(self):
        print("is torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
        print("device: {}".format(self.device))

        Clustering.angle_distance_threshold_degrees = self.config["angle_distance_threshold_degrees"]
        Clustering.recompute(math.sqrt(self.config["singular_value_quantil"]))

        self.log()
        self.scene_info = SceneInfo.read_scene(scene_name=self.config["scene_name"], type=self.config["scene_type"], file_name_suffix=self.file_name_suffix)
        self.setup_descriptor()
        if self.config[CartesianConfig.affnet_clustering]:
            self.dense_affnet = DenseAffNet(True, self.device)
            assert self.config["rectify_affine_affnet"], "'affnet_clustering' without 'rectify_affine_affnet' doesn't work"

        if self.config["rectify_affine_affnet"]:
            assert isinstance(self.feature_descriptor, HardNetDescriptor), "rectify_affine_affnet on, but without HardNet descriptor"

        scene_length = len(self.scene_info.img_pairs_lists)
        scene_length_range = range(0, scene_length)
        if self.matching_pairs is not None:
            self.matching_difficulties = scene_length_range

        self.depth_reader = DepthReader(self.scene_info.depth_input_dir())

        intersection = set(self.matching_difficulties).intersection(set(scene_length_range))
        self.matching_difficulties = list(intersection)

    def log(self):
        print("Pipeline config:")
        attr_list = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__") and attr not in ["stats", "stats_map"]]
        for attr_name in attr_list:
            if attr_name in ["scene_info"]:
                continue
            print("\t{}\t{}".format(attr_name, getattr(self, attr_name)))
        print()

        Config.log()
        Clustering.log()

    def get_img_processing_dir(self):
        return "{}/imgs/{}".format(self.output_dir, self.cache_map[Property.cache_img_data])

    def get_and_create_img_processing_dir(self):
        img_processing_dir = self.get_img_processing_dir()
        if not os.path.exists(img_processing_dir):
            Path(img_processing_dir).mkdir(parents=True, exist_ok=True)
        return img_processing_dir

    def get_cached_image_data_or_none(self, img_name, img, real_K):

        Timer.start_check_point("Serving img data from cache")
        img_processing_dir = self.get_and_create_img_processing_dir()
        img_data_path = "{}/{}_img_data.pkl".format(img_processing_dir, img_name)
        if self.use_cached_img_data and os.path.isfile(img_data_path):
            Timer.start_check_point("reading img processing data")
            with open(img_data_path, "rb") as f:
                print("img data for {} already computed, reading: {}".format(img_name, img_data_path))
                img_serialized_data: ImageSerializedData = pickle.load(f)
            Timer.end_check_point("reading img processing data")
            ret = ImageData.from_serialized_data(img=img,
                                                  real_K=real_K,
                                                  img_serialized_data=img_serialized_data)
        else:
            ret = None
        Timer.end_check_point("Serving img data from cache")
        return ret

    @staticmethod
    def save_img_data(img_data, img_data_path, img_name):
        Timer.start_check_point("saving img data")
        with open(img_data_path, "wb") as f:
            print("img data for {} saving into: {}".format(img_name, img_data_path))
            pickle.dump(img_data.to_serialized_data(), f)
        Timer.end_check_point("saving img data")

    def read_img(self, img_name):

        def format_to_read_mode(read_mode, img):
            if read_mode is None:
                return img
            if read_mode == "RGB":
                cv_mode = cv.COLOR_BGR2RGB
            elif read_mode == "GRAY":
                cv_mode = cv.COLOR_BGR2GRAY
            else:
                raise Exception("Unexpected value for read_mode: {}".format(read_mode))
            return cv.cvtColor(img, cv_mode)

        def scale_img(max_size, img):
            if max_size is None:
                return img
            if max_size < np.max(img.shape):
                resize_factor = np.max(img.shape) / max_size
                img = cv.resize(img, (int(img.shape[1] / resize_factor), int(img.shape[0] / resize_factor)), interpolation=cv.INTER_LINEAR)
            return img

        read_mode = self.config["img_read_mode"]
        max_size = self.config["img_max_size"]

        img_file_path = self.scene_info.get_img_file_path(img_name)
        img = cv.imread(img_file_path, None)
        if img is None:
            raise ValueError("img not found at {}".format(img_file_path))
        img = format_to_read_mode(read_mode, img)
        img = scale_img(max_size, img)

        if self.show_input_img:
            plt.figure(figsize=(9, 9))
            plt.title(img_name)
            plt.imshow(img)
            show_or_close(True)

        return img

    def possibly_set_custom_normals(self, img_name, focal_length, orig_height, orig_width, real_K):

        if self.config[CartesianConfig.affnet_custom_depth_lafs]:

            Timer.start_check_point("Custom normals")
            # get depth data - normals - lafs
            depth_map = self.depth_reader.read_depth_data(img_name)
            normals, _ = compute_normals_from_svd(focal_length, orig_height, orig_width, depth_map, device=torch.device('cpu'),
                                                         svd_weighted=self.config["svd_weighted"], svd_weighted_sigma=self.config["svd_weighted_sigma"])
            self.feature_descriptor.set_normals_np(normals, real_K)
            Timer.end_check_point("Custom normals")

    @timer_label_decorator("sky mask")
    def get_potential_sky_mask(self, img, h, w):
        if self.config[CartesianConfig.filter_sky]:
            non_sky_mask = get_nonsky_mask(img, h, w, self.device == torch.device("cuda"))
        else:
            non_sky_mask = np.ones((h, w), dtype=bool)
        return non_sky_mask

    @timer_label_decorator()
    def cluster_normals(self, normals, filter_mask):
        normals_deviced = normals.to(self.device)
        print("normals_deviced.device: {}".format(normals_deviced.device))
        normals_clusters_repr, normal_indices, valid_normals = cluster_normals(normals_deviced,
                                                                               filter_mask=filter_mask,
                                                                               mean_shift_type=self.config["mean_shift_type"],
                                                                               device=self.device,
                                                                               handle_antipodal_points=self.config["handle_antipodal_points"])
        return normals_clusters_repr, normal_indices, valid_normals

    @timer_label_decorator()
    def possibly_upsample_early(self, img, normal_indices):
        if self.upsample_early:
            return possibly_upsample_normals(img, normal_indices)
        else:
            return normal_indices

    @timer_label_decorator()
    def filter_valid_normals(self, normals):
        # LAST_MINUTE !!!
        valid_normal_indices = []
        for i, normal in enumerate(normals):
            angle_rad = math.acos(np.dot(normal, np.array([0, 0, -1])))
            angle_degrees = angle_rad * 180 / math.pi
            # print("angle: {} vs. angle threshold: {}".format(angle_degrees, Config.plane_threshold_degrees))
            if angle_degrees >= Config.plane_threshold_degrees:
                print("WARNING: too sharp of an angle with the -z axis, skipping the rectification")
                continue
            else:
                print("angle ok")
                valid_normal_indices.append(i)

        return valid_normal_indices

    @timer_label_decorator()
    def possibly_upsample_late(self, components_indices, img):
        if not self.upsample_early:
            assert np.all(components_indices < 256), "could not retype to np.uint8"
            components_indices = components_indices.astype(dtype=np.uint8)
            components_indices = possibly_upsample_normals(img, components_indices)
            components_indices = components_indices.astype(dtype=np.uint32)
        return components_indices

    @timer_label_decorator(tags=[COMPLETE_IMAGE_PAIR_MATCHING_TAG])
    def process_image(self, img_name, order):

        print("Processing: {}".format(img_name))
        img_processing_dir = self.get_and_create_img_processing_dir()
        Path(img_processing_dir).mkdir(parents=True, exist_ok=True)

        img = self.read_img(img_name)

        orig_height = img.shape[0]
        orig_width = img.shape[1]
        real_K = self.scene_info.get_img_K(img_name, img)
        if self.estimate_k:
            focal_length = (orig_width + orig_height) * self.focal_point_mean_factor
            K_for_rectification = np.array([
                [focal_length, 0,            orig_width / 2.0],
                [0,            focal_length, orig_height / 2.0],
                [0,            0,            1]
            ])
        else:
            K_for_rectification = real_K
            focal_length = real_K[0, 0]

        img_data_path = "{}/{}_img_data.pkl".format(img_processing_dir, img_name)
        cached_img_data = self.get_cached_image_data_or_none(img_name, img, real_K)
        if cached_img_data is not None:
            return cached_img_data

        self.possibly_set_custom_normals(img_name, focal_length, orig_height, orig_width, real_K)

        # without rectification at all
        if not self.config[CartesianConfig.rectify]:

            Timer.start_check_point("processing img without rectification")

            kps, descs = self.feature_descriptor.detectAndCompute(img, None)

            for kp in kps:
                kp.response = 1 * kp.response
                kp.size = 5 * kp.size

            ## NOTE: to visualize kpts without rectification

            # import torch.nn as nn
            # def upsample(data_2d, size_t):
            #     data_2d = torch.from_numpy(data_2d)
            #     data_2d = data_2d[None].permute(0, 3, 1, 2)
            #     upsample = nn.Upsample(size=size_t, mode='nearest')
            #     upsampled = upsample(data_2d)
            #     return upsampled.permute(0, 2, 3, 1)[0].numpy()
            #
            # # cp_img = img.copy()
            # cp_img = plt.imread("demos/work/frame_out.png")[:, :, :3]
            # cp_img = upsample(cp_img, (img.shape[:2])) * 255
            # cp_img = cp_img.astype(np.uint8).copy()
            #
            # cv.drawKeypoints(cp_img, kps, cp_img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # fig, ax = plt.subplots(figsize=(10, 10))
            # #fig = plt.figure(figsize=(20, 20))
            # #plt.title("New visos")
            # ax.set_axis_off()
            #
            # plt.imshow(cp_img)
            # plt.savefig("work/keypoints_new.png", bbox_inches='tight', pad_inches=0)
            # plt.show()

            img_data = ImageData(img=img,
                             real_K=real_K,
                             key_points=kps,
                             descriptions=descs,
                             normals=None,
                             ts_phis=None,
                             components_indices=None,
                             valid_components_dict=None)

            Pipeline.save_img_data(img_data, img_data_path, img_name)

            Timer.end_check_point("processing img without rectification")
            return img_data

        # rectify
        else:

            Timer.start_check_point(PROCESSING_IMG_FROM_SCRATCH_TAG, tags=["main"])

            rectify_affine_affnet = self.config["rectify_affine_affnet"]
            # NOTE 'affnet_no_clustering' means to (possibly rectify affine affnet but don't use clustering)
            affnet_no_clustering = self.config["affnet_no_clustering"]
            if rectify_affine_affnet and affnet_no_clustering:
                print("no prior clustering")
                img_data = ImageData(img=img,
                                     real_K=real_K,
                                     key_points=None,
                                     descriptions=None,
                                     normals=None,
                                     ts_phis=None,
                                     components_indices=None,
                                     valid_components_dict=None)
            else:

                if self.config[CartesianConfig.affnet_clustering]:
                    # (img, dense_affnet, conf, upsample_early):
                    img_data = affnet_clustering(img, img_name, self.dense_affnet, self.config, self.upsample_early, self.device == torch.device("cuda"))
                    img_data.real_K = real_K
                else:

                    depth_data = self.depth_reader.read_depth_data(img_name)
                    normals, s_values = compute_normals_from_svd(focal_length, orig_height, orig_width, depth_data, device=torch.device('cpu'),
                                                                 svd_weighted=self.config["svd_weighted"], svd_weighted_sigma=self.config["svd_weighted_sigma"])

                    non_sky_mask = self.get_potential_sky_mask(img, *normals.shape[:2])

                    filter_mask = self.apply_quantil_mask(img, img_name, depth_data, s_values, non_sky_mask)

                    normals_clusters_repr, normal_indices, valid_normals = self.cluster_normals(normals, filter_mask)

                    self.update_normals_stats(normal_indices, normals_clusters_repr, valid_normals, self.cache_map[Property.all_combinations], img_name)

                    # to be redesigned and removed
                    depth_data_file_name = "{}.npy".format(img_name)
                    show_or_save_clusters(normals,
                                          normal_indices,
                                          normals_clusters_repr,
                                          img_processing_dir,
                                          depth_data_file_name,
                                          show=self.show_clusters,
                                          save=self.save_clusters)

                    normal_indices = self.possibly_upsample_early(img, normal_indices)

                    valid_normal_indices = self.filter_valid_normals(normals_clusters_repr)

                    components_indices, valid_components_dict = get_connected_components(normal_indices, valid_normal_indices,
                                                                                         closing_size=self.connected_components_closing_size,
                                                                                         flood_filling=self.connected_components_flood_fill,
                                                                                         connectivity=self.connected_components_connectivity)
                    components_indices = self.possibly_upsample_late(components_indices, img)

                    components_out_path = "{}/{}_cluster_connected_components".format(img_processing_dir, img_name)
                    get_and_show_components(components_indices,
                                            valid_components_dict,
                                            normals=normals_clusters_repr,
                                            show=self.show_clustered_components,
                                            save=self.save_clustered_components,
                                            path=components_out_path)

                    img_data = ImageData(img=img,
                                         real_K=real_K,
                                         key_points=None,
                                         descriptions=None,
                                         normals=normals_clusters_repr,
                                         ts_phis=None,
                                         components_indices=components_indices,
                                         valid_components_dict=valid_components_dict)


            fixed_rot_condition = self.config["rectify_by_fixed_rotation"]
            finish_cond = self.get_stage_number() <= self.stages_map["before_rectification"]
            if fixed_rot_condition or finish_cond:
                print("rectify_by_fixed_rotation: {}, finish before_rectification: {}".format(fixed_rot_condition, finish_cond))
                print("process_image done")

            elif self.config["rectify_affine_affnet"]:

                # FIXME fail fast!!!
                assert isinstance(self.feature_descriptor, HardNetDescriptor), "rectify_affine_affnet on, but without HardNet descriptor"

                # TODO ideally remove this hack
                # NOTE that unfortunately it is not very easy to do - hopefully the experiments will show this option
                # won't be successful as it is not the most natural way to do IMHO
                conf_to_use = self.config
                if self.config[CartesianConfig.affnet_clustering]:
                    conf_to_use = self.config.copy()
                    conf_to_use["affnet_covering_max_iter"] = 1
                    print()

                kpts_struct = affnet_rectify(img_name,
                                             self.feature_descriptor,
                                             img_data,
                                             conf_to_use,
                                             device=self.device,
                                             params_key=self.cache_map[Property.all_combinations],
                                             stats_map=self.stats)
                img_data.key_points = kpts_struct.kps
                img_data.descriptions = kpts_struct.descs

            else:

                key = "rotation_alpha1" if order == 0 else "rotation_alpha2"
                rotation_factor = self.config[key]

                # get rectification
                rectification_path_prefix = "{}/{}".format(img_processing_dir, img_name)

                img_data.key_points, img_data.descriptions, unrectified_indices = get_rectified_keypoints(
                    normals_clusters_repr,
                    components_indices,
                    valid_components_dict,
                    img,
                    K_for_rectification,
                    descriptor=self.feature_descriptor,
                    img_name=img_name,
                    clip_angle=self.clip_angle,
                    show=self.show_rectification,
                    save=self.save_rectification,
                    out_prefix=rectification_path_prefix,
                    rotation_factor=rotation_factor,
                    all_unrectified=self.config[CartesianConfig.all_unrectified],
                    params_key=self.cache_map[Property.all_combinations],
                    stats_map=self.stats,
                    )

            Timer.end_check_point(PROCESSING_IMG_FROM_SCRATCH_TAG)

            Pipeline.save_img_data(img_data, img_data_path, img_name)

            return img_data

    def get_nonsky_mask_possibly_cached(self, img_name, np_image, h_w):
        height, width = h_w
        Timer.start_check_point("sky_mask")
        sky_cache_dir = "{}/cache/sky".format(self.output_dir)
        if not os.path.exists(sky_cache_dir):
            Path(sky_cache_dir).mkdir(parents=True, exist_ok=True)
        sky_cache_file = "{}/{}.npy".format(sky_cache_dir, img_name)
        if os.path.isfile(sky_cache_file):
            with open(sky_cache_file, "rb") as f:
                ret = pickle.load(f)
        else:
            ret = get_nonsky_mask(np_image, height, width, self.device == torch.device("cuda"))
            with open(sky_cache_file, "wb") as f:
                pickle.dump(ret, f)
        Timer.end_check_point("sky_mask")
        return ret

    def compute_normals_possibly_cached(self,
            img_name,
            focal_length,
            orig_height,
            orig_width,
            depth_data,
            simple_weighing=True,
            smaller_window=False,
            device=torch.device('cpu'),
            use_cache=True,
            svd_weighted_sigma=0.8
    ):

        Timer.start_check_point("compute_normals")
        cache_file = None
        ret = None
        if use_cache:
            cache_dir = "{}/cache/normals".format(self.output_dir)
            if not os.path.exists(cache_dir):
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache_file = "{}/{}_{}.npy".format(cache_dir, img_name, svd_weighted_sigma)
            if os.path.isfile(cache_file):
                with open(cache_file, "rb") as f:
                    ret = pickle.load(f)
        if ret is None:
            ret = compute_normals_from_svd(focal_length, orig_height, orig_width, depth_data,
                                           simple_weighing, smaller_window, device, svd_weighted_sigma=svd_weighted_sigma)
            if use_cache:
                with open(cache_file, "wb") as f:
                    pickle.dump(ret, f)
        Timer.end_check_point("compute_normals")
        return ret

    @timer_label_decorator()
    def update_normals_stats(self, normal_indices, normals_clusters_repr, valid_normals, params_key, img_name):

        sums = np.array([np.sum(normal_indices == i) for i in range(len(normals_clusters_repr))])
        indices = np.argsort(-sums)

        # eventually delete this check and the two lines above
        for i in range(len(indices)):
            if i != indices[i]:
                print("Warning: i != indices[i]")

        normals_clusters_repr_sorted = normals_clusters_repr[indices]

        degrees_list = get_degrees_between_normals(normals_clusters_repr_sorted)
        print("pipeline:compute_img_normals:normals_clusters_repr_sorted: {}".format(normals_clusters_repr_sorted))
        print("pipeline:compute_img_normals:degrees_list: {}".format(degrees_list))
        self.update_stats_map(["normals_degrees", params_key, img_name], degrees_list)
        self.update_stats_map(["normals", params_key, img_name], normals_clusters_repr)
        self.update_stats_map(["valid_normals", params_key, img_name], valid_normals)

    @timer_label_decorator("quantil_mask")
    def apply_quantil_mask(self, img, img_name, depth_data, s_values, non_sky_mask):
        quantil_mask = self.get_quantil_mask(img, img_name, self.config["singular_value_quantil"], depth_data[2:4],
                                             s_values, depth_data, non_sky_mask)
        filter_mask = non_sky_mask & quantil_mask
        return filter_mask

    def get_quantil_mask(self, img, img_name, singular_value_quantil, h_w_size, s_values, depth_data, sky_mask_np):
        if singular_value_quantil == 1.0: # and False: NOTE #visualizations
            # NOTE this works by accident as h_w_size is not a tuple with dimensions, but Tensor(0, 1, h, w),
            # which gives mask = array(True) (!!)
            mask = np.ones(h_w_size, dtype=bool)
        else:
            s_values_ratio = True
            if s_values_ratio:
                singular_values_order = s_values[:, :, 2] / torch.clamp(s_values[:, :, 1], min=1e-19)
            else:
                singular_values_order = s_values[:, :, 2]
                singular_values_order = singular_values_order / torch.clamp(depth_data[0, 0], min=1e-19)
            singular_values_order = singular_values_order + (1 - sky_mask_np) * singular_values_order.max().item()

            h, w = singular_values_order.shape[0], singular_values_order.shape[1]
            singular_values_order = singular_values_order.reshape(h * w)
            _, indices = torch.sort(singular_values_order)

            mask = torch.zeros_like(singular_values_order, dtype=torch.bool)
            non_sky = sky_mask_np.sum()
            mask[indices[:int(non_sky * singular_value_quantil)]] = True
            mask = mask.reshape(h, w).numpy()

            show_sky_mask(img, mask, img_name, show=self.show_sky_mask, save=False, title="quantile mask")
            show_sky_mask(img, sky_mask_np & mask, img_name, show=self.show_sky_mask, save=False, title="quantile and sky mask")
            # NOTE this is just to get the #visualization
            # show_sky_mask(img, mask, "{}_sky_sv_mask.png".format(img_name), show=self.show_sky_mask, save=False, title="quantile mask")
            # show_sky_mask(img, sky_mask_np & mask, "{}_sky_mask.png".format(img_name), show=self.show_sky_mask, save=False, title="quantile and sky mask")

        return mask

    def compute_img_normals(self, img, img_name, stats_key_prefix=None, use_normals_cache=True):

        print("processing img {}".format(img_name))

        # get focal_length
        orig_height = img.shape[0]
        orig_width = img.shape[1]
        if self.estimate_k:
            focal_length = (orig_width + orig_height) * self.focal_point_mean_factor
        else:
            real_K = self.scene_info.get_img_K(img_name, img)
            focal_length = real_K[0, 0]
            assert abs(real_K[0, 2] * 2 - orig_width) < 0.5
            assert abs(real_K[1, 2] * 2 - orig_height) < 0.5

        depth_data = self.depth_reader.read_depth_data(img_name)
        print("depth_data device {}".format(depth_data.device))

        img_processing_dir = self.get_and_create_img_processing_dir()
        sky_out_path = "{}/{}_sky_mask.jpg".format(img_processing_dir, img_name)
        # filter_mask is a numpy array
        sky_mask_np = self.get_nonsky_mask_possibly_cached(img_name, img, depth_data.shape[2:4])
        show_sky_mask(img, sky_mask_np, img_name, show=self.show_sky_mask, save=self.save_sky_mask, path=sky_out_path)

        counter = 0
        for sigma in [0.8, 1.2, 1.6]:

            orig_normals, s_values = self.compute_normals_possibly_cached(img_name,
                                                                          focal_length,
                                                                          orig_height,
                                                                          orig_width,
                                                                          depth_data,
                                                                          simple_weighing=True,
                                                                          smaller_window=(sigma == 0.0),
                                                                          device=torch.device('cpu'),
                                                                          use_cache=use_normals_cache,
                                                                          svd_weighted_sigma=sigma)

            print("orig_normals.device: {}".format(orig_normals.device))

            for normal_sigma in [None]:

                if normal_sigma is None:
                    normals = orig_normals
                else:
                    bf_key = "bilateral_filter_{}".format(normal_sigma)
                    Timer.start_check_point(bf_key)
                    normals = bilateral_filter(orig_normals, filter_mask=sky_mask_np, normal_sigma=normal_sigma, device=torch.device('cpu'))
                    print("normals.device after bilateral_filter: {}".format(normals.device))
                    Timer.end_check_point(bf_key)

                for mean_shift in ["full", "mean", None]:

                    for singular_value_quantil in [0.6, 0.8, 1.0]:

                        for angle_distance_threshold_degrees in [5, 10, 15, 20, 25, 30, 35]:

                            for adaptive in [False]:

                                if adaptive is True and mean_shift != "full":
                                    continue

                                compute = False
                                # if mean_shift is None and singular_value_quantil == 1.0 and angle_distance_threshold_degrees == 25 and sigma == 0.8:
                                #     compute = True
                                # if mean_shift in ["mean", "full"] and angle_distance_threshold_degrees == 35 and sigma == 0.8:
                                #     compute = True
                                if mean_shift in ["mean"] and angle_distance_threshold_degrees in [25] and sigma == 0.8 and singular_value_quantil in [0.8, 1.0]:
                                    compute = True
                                if not compute:
                                    continue

                                counter = counter + 1

                                ms_str = "ms_{}".format(mean_shift)
                                params_key = "{}_{}_{}_{}_{}_{}".format(ms_str, singular_value_quantil, angle_distance_threshold_degrees, sigma, normal_sigma, adaptive)
                                if stats_key_prefix is not None:
                                    params_key = "{}_{}".format(stats_key_prefix, params_key)

                                print("params_key: {}".format(params_key))
                                print("Params: s_value_hist_ratio: {}, "
                                      "angle_distance_threshold_degrees: {}, "
                                      "svd sigma: {}, "
                                      "mean shift step: {},"
                                      "normal sigma: {},"
                                      "adaptive: {}"
                                      .format(singular_value_quantil, angle_distance_threshold_degrees, sigma, ms_str, normal_sigma, adaptive))

                                Clustering.angle_distance_threshold_degrees = angle_distance_threshold_degrees
                                Clustering.recompute(math.sqrt(singular_value_quantil))

                                quantil_mask = self.get_quantil_mask(img, img_name, singular_value_quantil, depth_data[2:4], s_values, depth_data, sky_mask_np)
                                filter_mask = sky_mask_np & quantil_mask

                                cp_key = "clustering_{}_{}".format(mean_shift, adaptive)
                                Timer.start_check_point(cp_key)
                                normals_deviced = normals.to(self.device)
                                print("normals_deviced.device: {}".format(normals_deviced.device))
                                normals_clusters_repr, normal_indices, valid_normals = cluster_normals(normals_deviced,
                                                                                                       filter_mask=filter_mask,
                                                                                                       mean_shift_type=mean_shift,
                                                                                                       adaptive=adaptive,
                                                                                                       return_all=True,
                                                                                                       device=self.device,
                                                                                                       handle_antipodal_points=True)

                                Timer.end_check_point(cp_key)
                                self.update_normals_stats(normal_indices, normals_clusters_repr, valid_normals, params_key, img_name)

                                show_or_save_clusters(normals,
                                                      normal_indices,
                                                      normals_clusters_repr,
                                                      img_processing_dir,
                                                      "{}: {} ".format(img_name, params_key),
                                                      show=self.show_clusters,
                                                      save=self.save_clusters)
        print("counter: {}".format(counter))

    def update_stats_map(self, key_list, obj):
        update_stats_map_static(key_list, obj, self.stats)
        # map = ensure_keys(self.stats, key_list[:-1])
        # map[key_list[-1]] = obj

    def run_sequential_pipeline(self):

        assert "sequential_files_limit" in self.config or self.chosen_depth_files is not None

        self.start()

        file_names, _ = self.scene_info.get_megadepth_file_names_and_dir(None, self.chosen_depth_files)
        if self.chosen_depth_files is None:
            file_names_permuted = [file_names[two_hundred_permutation[i]] for i in range(self.config["sequential_files_limit"])]
        else:
            file_names_permuted = file_names

        for idx, depth_data_file_name in enumerate(file_names_permuted):
            self.process_image(depth_data_file_name[:-4], idx % 2)

        evaluate_normals_stats(self.stats)
        self.save_stats("sequential")

    def show_and_read_img_from_path(self, img_name, img_file_path):
        img = cv.imread(img_file_path, None)
        if self.show_input_img:
            plt.figure(figsize=(9, 9))
            plt.title(img_name)
            plt.imshow(img)
            show_or_close(True)
        return img

    def show_and_read_img(self, img_name):
        img_file_path = self.scene_info.get_img_file_path(img_name)
        return self.show_and_read_img_from_path(img_name, img_file_path)

    def run(self):
        if self.method == "compute_normals":
            self.run_sequential_for_normals()
        elif self.method == "compute_normals_compare":
            self.run_sequential_for_normals_compare()
        elif self.method == "run_sequential_pipeline":
            self.run_sequential_pipeline()
        elif self.method == "run_matching_pipeline":
            self.run_matching_pipeline()
        else:
            print("incorrect value of '{}' for method. Choose one from 'compute_normals', 'run_sequential_pipeline' or 'run_matching_pipeline'".format(self.method))

    # TODO deprecated
    def run_sequential_for_normals_compare(self):

        self.start()

        globs_and_prefixes = [("depth_data/megadepth_ds/depths/*_orig.npy", "gt"),
                              ("depth_data/megadepth_ds/depths/*_o.npy", "estimated")]

        depth_reader = DepthReader("depth_data/megadepth_ds/depths")
        imgs_base = "depth_data/megadepth_ds/imgs"

        for glob_str, prefix in globs_and_prefixes:

            gl = glob.glob(glob_str)
            if gl is None or len(gl) == 0:
                raise Exception("nothing under {}".format(glob_str))

            file_names = [path.split("/")[-1] for path in gl]
            print("file_names: {}".format(file_names))

            for focal_point_mean_factor in [0.47]: #5, 0.5, 0.55]:
                self.focal_point_mean_factor = focal_point_mean_factor

                for i, depth_data_file_name in enumerate(file_names):
                    # if not depth_data_file_name.__contains__("101426383_d1d108b4c5"):
                    #     continue
                    if not depth_data_file_name.__contains__("110724086_954bbd1a5e"):
                        continue

                    img_name = depth_data_file_name[:-4]
                    img_name_img = img_name[:-5] if img_name.endswith("orig") else img_name
                    img_path = "{}/{}.jpg".format(imgs_base, img_name_img)
                    img = self.show_and_read_img_from_path(img_name, img_path)

                    if prefix == "gt":
                        sky_cache_dir = "{}/cache/sky".format(self.output_dir)
                        if not os.path.exists(sky_cache_dir):
                            Path(sky_cache_dir).mkdir(parents=True, exist_ok=True)
                        sky_cache_files = ["{}/{}.npy".format(sky_cache_dir, img_name), "{}/{}.npy".format(sky_cache_dir, img_name[:-5])]
                        depth_data = depth_reader.read_depth_data(depth_data_file_name[:-4])
                        sky_mask = (depth_data[0, 0] != 0.0).to(torch.int).numpy()
                        for sky_cache_file in sky_cache_files:
                            with open(sky_cache_file, "wb") as f:
                                pickle.dump(sky_mask, f)

                    prefix_full = "{}_{}".format(prefix, focal_point_mean_factor)
                    self.compute_img_normals(img, img_name, prefix_full, use_normals_cache=False)
                    if i % 10 == 0:
                        self.save_stats("normals_{}".format(i))
                    evaluate_normals_stats(self.stats)
                    Timer.log_stats()

        self.save_stats("normals")

    # TODO deprecated
    def run_sequential_for_normals(self):

        self.start()

        file_names, _ = self.scene_info.get_megadepth_file_names_and_dir(None, self.chosen_depth_files)
        file_names_permuted = [file_names[two_hundred_permutation[i]] for i in range(self.config["sequential_files_limit"])]
        for i, depth_data_file_name in enumerate(file_names_permuted):
            img_name = depth_data_file_name[:-4]
            img = self.show_and_read_img(img_name)
            self.compute_img_normals(img, img_name)
            if i % 10 == 0:
                self.save_stats("normals_{}".format(i))
            evaluate_normals_stats(self.stats)
            Timer.log_stats()

        self.save_stats("normals")

    def update_matching_stats(self, key, difficulty, img_pair_name, stats_struct: Stats):
        self.update_stats_map(["matching", key, difficulty, img_pair_name, "kps1"], stats_struct.all_features_1)
        self.update_stats_map(["matching", key, difficulty, img_pair_name, "kps2"], stats_struct.all_features_2)
        self.update_stats_map(["matching", key, difficulty, img_pair_name, "tentatives"], stats_struct.tentative_matches)
        self.update_stats_map(["matching", key, difficulty, img_pair_name, "inliers"], stats_struct.inliers)

    def get_tex_file_name(self, difficulty):
        tex_dir = "{}/tex".format(self.output_dir)
        if not os.path.exists(tex_dir):
            Path(tex_dir).mkdir(parents=True, exist_ok=True)
        return "{}/tex_after_{}_{}".format(tex_dir, self.cache_map[Property.all_combinations], difficulty)

    def get_diff_stats_file(self, difficulty=None):
        diff_str = "all" if difficulty is None else str(difficulty)
        diff_stats_dir = "{}/stats/{}".format(self.output_dir, self.cache_map[Property.all_combinations])
        if not os.path.exists(diff_stats_dir):
            Path(diff_stats_dir).mkdir(parents=True, exist_ok=True)
        return "{}/stats_diff_{}.pkl".format(diff_stats_dir, diff_str)

    def get_stats_key(self):
        return self.cache_map[Property.all_combinations]

    def ensure_stats_key(self, map, suffix=""):
        key = self.get_stats_key() + suffix
        if not map.__contains__(key):
            map[key] = {}

    def do_matching(self, image_data_list, img_pair, matching_out_dir, stats_map_diff, difficulty):

        if self.planes_based_matching:
            E, inlier_mask, src_pts, dst_pts, tentative_matches = match_images_with_dominant_planes(
                image_data_list[0],
                image_data_list[1],
                use_degensac=self.use_degensac,
                find_fundamental=self.estimate_k,
                img_pair=img_pair,
                out_dir=matching_out_dir,
                show=self.show_matching,
                save=self.save_matching,
                ratio_thresh=self.knn_ratio_threshold,
                ransac_th=self.ransac_th,
                ransac_conf=self.ransac_conf,
                ransac_iters=self.ransac_iters
            )

        elif self.use_degensac:
            E, inlier_mask, src_pts, dst_pts, tentative_matches = match_find_F_degensac(
                image_data_list[0].img,
                image_data_list[0].key_points,
                image_data_list[0].descriptions,
                image_data_list[0].real_K,
                image_data_list[1].img,
                image_data_list[1].key_points,
                image_data_list[1].descriptions,
                image_data_list[1].real_K,
                img_pair,
                matching_out_dir,
                show=self.show_matching,
                save=self.save_matching,
                ratio_thresh=self.knn_ratio_threshold,
                ransac_th=self.ransac_th,
                ransac_conf=self.ransac_conf,
                ransac_iters=self.ransac_iters
            )

        else:
            # NOTE using img_datax.real_K for a call to findE
            E, inlier_mask, src_pts, dst_pts, tentative_matches = match_epipolar(
                image_data_list[0], image_data_list[1],
                find_fundamental=self.estimate_k,
                img_pair=img_pair,
                out_dir=matching_out_dir,
                show=self.show_matching,
                save=self.save_matching,
                ratio_thresh=self.knn_ratio_threshold,
                ransac_th=self.ransac_th,
                ransac_conf=self.ransac_conf,
                ransac_iters=self.ransac_iters,
                cfg=self.config,
            )

            if E is None:
                ValueError("E is None")

        stats_struct = evaluate_matching(self.scene_info,
                                         E,
                                         image_data_list,
                                         tentative_matches,
                                         inlier_mask,
                                         img_pair,
                                         stats_map_diff,
                                         ransac_th=self.ransac_th
                                         )

        self.update_matching_stats(self.cache_map[Property.all_combinations], difficulty,
                                   "{}_{}".format(img_pair.img1, img_pair.img2), stats_struct)

    def estimate_rotation_via_normals(self, normals1, normals2, img_pair, pair_key, zero_around_z):

        # get R
        normals1 = possibly_expand_normals(normals1)
        normals2 = possibly_expand_normals(normals2)
        solutions = find_sorted_rotations(normals1, normals2, zero_around_z)

        r_vec_first = solutions[0].rotation_vector
        r_matrix_first = get_rotation_matrix_safe(r_vec_first)
        first_err = compare_R_to_GT(img_pair, self.scene_info, r_matrix_first)
        GT_err = compare_R_to_GT(img_pair, self.scene_info, np.eye(3))
        GT_mat, _ = get_GT_R_t(img_pair, self.scene_info)
        GT_vec = KG.rotation_matrix_to_angle_axis(torch.from_numpy(GT_mat)[None]).detach().cpu().numpy()[0]
        print("normals counts: ({}, {})".format(normals1.shape[0], normals2.shape[0]))
        print("estimate_rotation_via_normals: I error against GT: {}".format(GT_err))
        print("estimate_rotation_via_normals: error against GT: {}".format(first_err))
        print("estimate_rotation_via_normals: objective function value: {}".format(solutions[0].objective_fnc))
        print("estimate_rotation_via_normals: rotation vector GT: {}".format(GT_vec))
        print("estimate_rotation_via_normals: rotation vector: {}".format(solutions[0].rotation_vector))

        top_5_err = first_err
        for solution in solutions[1:5]:
            r = get_rotation_matrix_safe(solution.rotation_vector)
            err_q = compare_R_to_GT(img_pair, self.scene_info, r)
            if err_q < top_5_err:
                top_5_err = err_q

        print("estimate_rotation_via_normals: top_5_err: {}".format(top_5_err))

        return r_vec_first

    def rectify_by_fixed_rotation_update(self, img_data, r, img_name):

        if self.estimate_k:
            raise NotImplemented("img_data.real_K != K_for_rectification")

        # get rectification
        rectification_path_prefix = "{}/{}".format(self.get_img_processing_dir(), img_name)
        kps, descs, _ = get_rectified_keypoints(img_data.normals,
                                                img_data.components_indices,
                                                img_data.valid_components_dict,
                                                img_data.img,
                                                img_data.real_K,  # K_for_rectification,
                                                descriptor=self.feature_descriptor,
                                                img_name=img_name,
                                                fixed_rotation_vector=r,
                                                clip_angle=self.clip_angle,
                                                show=self.show_rectification,
                                                save=self.save_rectification,
                                                out_prefix=rectification_path_prefix,
                                                all_unrectified=self.config[CartesianConfig.all_unrectified],
                                                params_key=self.cache_map[Property.all_combinations],
                                                stats_map=self.stats,
                                                )

        img_data.key_points = kps
        img_data.descriptions = descs
        return img_data

    def find_cached_pair_update_stat(self, img_pair: ImagePairEntry, difficulty):
        s_map = self.stats_map[self.get_stats_key()]
        key = SceneInfo.get_key_from_pair(img_pair)
        for cached_diff in s_map:
            cached_diff_map = s_map[cached_diff]
            if cached_diff_map.__contains__(key):
                s_map[difficulty][key] = cached_diff_map[key]
                return True
        return False

    def run_matching_pipeline(self):

        self.start()

        self.ensure_stats_key(self.stats_map)

        stats_counter = 0

        for difficulty in self.matching_difficulties:
            print("Difficulty: {}".format(difficulty))

            stats_map_diff = {}
            self.stats_map[self.get_stats_key()][difficulty] = stats_map_diff

            processed_pairs = 0
            for img_pair in self.scene_info.img_pairs_lists[difficulty]:

                pair_key = SceneInfo.get_key(img_pair.img1, img_pair.img2)
                if self.matching_pairs is not None and pair_key not in self.matching_pairs:
                    continue

                if self.matching_pairs is None and self.matching_limit is not None and processed_pairs >= self.matching_limit:
                    print("Reached matching limit of {} for difficulty {}".format(self.matching_limit, difficulty))
                    break

                print("Will be matching pair {}".format(pair_key))
                stats_counter = stats_counter + 1

                if self.find_cached_pair_update_stat(img_pair, difficulty):
                    print("{} is cached, skipping the computation".format(SceneInfo.get_key_from_pair(img_pair)))
                else:
                    Timer.start_check_point(COMPLETE_IMAGE_PAIR_MATCHING_TAG, tags=["main", COMPLETE_IMAGE_PAIR_MATCHING_TAG])

                    try:

                        matching_out_dir = "{}/matching".format(self.output_dir)
                        Path(matching_out_dir).mkdir(parents=True, exist_ok=True)

                        # I might not need normals yet
                        # img1, K_1, kps1, descs1, normals1, components_indices1, valid_components_dict1

                        image_data = []
                        processed_img_index = 0
                        try:
                            for idx, img in enumerate([img_pair.img1, img_pair.img2]):
                                image_data.append(self.process_image(img, idx))
                                processed_img_index += 1
                        except:
                            print("(processing image) {}st/nd image from {}_{} couldn't be processed, skipping the matching pair".format(processed_img_index, img_pair.img1, img_pair.img2))
                            print(traceback.format_exc(), file=sys.stdout)
                            continue

                        rectify_affine_affnet = self.config["rectify_affine_affnet"]
                        affnet_no_clustering = self.config["affnet_no_clustering"]
                        if self.config[CartesianConfig.rectify] and (not rectify_affine_affnet or not affnet_no_clustering) and not self.config[CartesianConfig.affnet_clustering]:
                            zero_around_z = self.config["rectify_by_0_around_z"]
                            estimated_r_vec = self.estimate_rotation_via_normals(image_data[0].normals, image_data[1].normals, img_pair, pair_key, zero_around_z)

                        if self.config["rectify_by_fixed_rotation"]:

                            if self.config["rectify_by_GT"]:
                                GT_R, _ = get_GT_R_t(img_pair, self.scene_info)
                                r_vec_full = KG.rotation_matrix_to_angle_axis(torch.from_numpy(GT_R)[None]).detach().cpu().numpy()[0]
                            else:
                                r_vec_full = estimated_r_vec

                            for idx, image_data_item in enumerate(image_data):
                                if idx == 0:
                                    r_vec = r_vec_full / 2
                                    img_name = img_pair.img1
                                else:
                                    r_vec = -r_vec_full / 2
                                    img_name = img_pair.img2

                                self.rectify_by_fixed_rotation_update(image_data_item, r_vec, img_name)

                        if self.get_stage_number() >= self.stages_map["final"]:
                            self.do_matching(image_data, img_pair, matching_out_dir, stats_map_diff, difficulty)

                        processed_pairs = processed_pairs + 1

                    except:
                        print("(matching) {} couldn't be processed, skipping the matching pair".format(pair_key))
                        print(traceback.format_exc(), file=sys.stdout)

                    Timer.end_check_point(COMPLETE_IMAGE_PAIR_MATCHING_TAG)

                if stats_counter % 10 == 0:
                    evaluate_stats(self.stats)
                evaluate_all_matching_stats_even_normalized(self.stats_map)

                Timer.log_stats()

            if processed_pairs > 0:
                norm_scene_info = self.scene_info if self.matching_pairs is None else None
                evaluate_all_matching_stats_even_normalized(self.stats_map, tex_save_path_prefix=self.get_tex_file_name(difficulty), scene_info=norm_scene_info)
                evaluate_stats(self.stats)

            stats_file_name = self.get_diff_stats_file(difficulty)
            with open(stats_file_name, "wb") as f:
                pickle.dump(stats_map_diff, f)
            self.save_stats("matching_after_{}_diff_{}".format(self.cache_map[Property.all_combinations], difficulty))

        all_stats_file_name = self.get_diff_stats_file()
        with open(all_stats_file_name, "wb") as f:
            pickle.dump(self.stats_map, f)

        self.save_stats("matching_after_{}".format(self.cache_map[Property.all_combinations]))
        self.log()
        # These two are different approaches to stats
        evaluate_stats(self.stats)
        norm_scene_info = self.scene_info if self.matching_pairs is None else None
        evaluate_all_matching_stats_even_normalized(self.stats_map, tex_save_path_prefix=self.get_tex_file_name(100), scene_info=norm_scene_info)

    def save_stats(self, key=""):
        file_name = None
        try:
            file_name = "{}/stats_{}_{}.pkl".format(self.output_dir, key.replace("/", "_"), get_tmsp())
            with open(file_name, "wb") as f:
                pickle.dump(self.stats, f)
                print("stats saved")
        except:
            print("Stats under file name '{}' couldn't be saved".format(file_name))
            print(traceback.format_exc(), file=sys.stdout)


def get_tmsp():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S_%f")


def append_all(pipeline, str):
    use_degensac = "DEGENSAC" if pipeline.use_degensac else "RANSAC"
    estimate_K = "estimatedK" if pipeline.estimate_k else "GTK"
    timestamp = get_tmsp()
    return "{}_{}_{}_{}".format(str, use_degensac, estimate_K, timestamp)


def run_pipeline(args, config_file_path):

    pipeline, config_map = Pipeline.configure(config_file_path, args)
    all_configs = CartesianConfig.get_configs(config_map)
    print("first iterate through the configs:")
    for config, cache_map in all_configs:
        print("Config: {}".format(config))
        print("Cache map: {}".format(cache_map))

    print("now start the pipeline:")
    for config, cache_map in all_configs:
        Timer.start()
        pipeline.config = config
        pipeline.cache_map = cache_map
        pipeline.run()
        print("Log stats for {}:".format(pipeline.get_stats_key()))
        Timer.log_stats()
        print("{} finished".format(pipeline.get_stats_key()))

    print("process finished")


def main():

    parser = argparse.ArgumentParser(prog='pipeline')
    parser.add_argument('--output_dir', help='output dir')
    args = parser.parse_args()
    run_pipeline(args, "config.txt")


if __name__ == "__main__":
    main()
