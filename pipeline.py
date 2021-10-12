
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import cv2 as cv
import pickle
import traceback
import sys

import numpy as np
import torch
import argparse

from config import Config
from connected_components import get_connected_components, get_and_show_components
from depth_to_normals import compute_only_normals
from depth_to_normals import show_sky_mask, cluster_normals, show_or_save_clusters, read_depth_data, compute_normals_from_svd
from matching import match_epipolar, match_find_F_degensac, match_images_with_dominant_planes
from rectification import possibly_upsample_normals, get_rectified_keypoints
from scene_info import SceneInfo
from utils import Timer
from img_utils import show_or_close, get_degrees_between_normals
from evaluation import *
from sky_filter import get_nonsky_mask
from clustering import Clustering

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


@dataclass
class Pipeline:

    scene_name = None
    scene_type = None
    permutation_limit = None
    method = None
    file_name_suffix = None
    output_dir = None
    output_dir_prefix = None

    ransac_th = 0.5
    ransac_conf = 0.9999
    ransac_iters = 100000

    # actually unused
    show_save_normals = False
    show_orig_image = True

    # ! FIXME not really compatible with matching pairs
    chosen_depth_files = None
    sequential_files_limit = None

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

    rectify = True
    clip_angle = None

    knn_ratio_threshold = 0.85

    use_cached_img_data = True

    upsample_early = True

    # connected components
    connected_components_connectivity = 4
    connected_components_closing_size = None
    connected_components_flood_fill = False

    stats = {}

    @staticmethod
    def configure(config_file_name: str, args):

        # https://docs.python.org/2/library/configparser.html
        #     import configparser
        #     config = configparser.ConfigParser()
        #     config.read("config.txt")
        #     cf = config['settings']
        #     sc = cf['save_clusters']
        #     print()

        feature_descriptors_str_map = {
            "SIFT": cv.SIFT_create(),
        }

        pipeline = Pipeline()

        with open(config_file_name) as f:
            for line in f:

                if line.strip().startswith("#"):
                    continue
                elif line.strip() == "":
                    continue

                k, v = line.partition("=")[::2]
                k = k.strip()
                v = v.strip()

                if k == "scene_name":
                    pipeline.scene_name = v
                elif k == "permutation_limit":
                    pipeline.permutation_limit = int(v)
                elif k == "method":
                    pipeline.method = v
                elif k == "scene_type":
                    pipeline.scene_type = v
                elif k == "file_name_suffix":
                    pipeline.file_name_suffix = v
                elif k == "rectify":
                    pipeline.rectify = v.lower() == "true"
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
                elif k == "feature_descriptor":
                    pipeline.feature_descriptor = feature_descriptors_str_map[v]
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
                    print("WARNING - unrecognized param: {}".format(k))

        pipeline.matching_difficulties = list(range(matching_difficulties_min, matching_difficulties_max))

        if args is not None and args.__contains__("output_dir"):
            pipeline.output_dir = args.output_dir
        elif pipeline.output_dir is None:
            pipeline.output_dir = append_all(pipeline, pipeline.output_dir_prefix)

        assert not pipeline.planes_based_matching or pipeline.rectify, "rectification must be on for planes_based_matching"

        return pipeline

    def start(self):
        print("is torch.cuda.is_available(): {}".format(torch.cuda.is_available()))

        self.log()
        self.scene_info = SceneInfo.read_scene(self.scene_name, self.scene_type, file_name_suffix=self.file_name_suffix)

        scene_length = len(self.scene_info.img_pairs_lists)
        scene_length_range = range(0, scene_length)
        if self.matching_pairs is not None:
            self.matching_difficulties = scene_length_range

        self.depth_input_dir = self.scene_info.depth_input_dir()
        intersection = set(self.matching_difficulties).intersection(set(scene_length_range))
        self.matching_difficulties = list(intersection)


    def log(self):
        print("Pipeline config:")
        attr_list = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for attr_name in attr_list:
            if attr_name in ["scene_info"]:
                continue
            print("\t{}\t{}".format(attr_name, getattr(self, attr_name)))
        print()

        Config.log()
        Clustering.log()

    def process_image(self, img_name):

        if not self.stats.keys().__contains__("imgs_data"):
            self.stats["imgs_data"] = {}

        Timer.start_check_point("processing img")
        print("Processing: {}".format(img_name))
        img_processing_dir = "{}/imgs".format(self.output_dir)
        Path(img_processing_dir).mkdir(parents=True, exist_ok=True)

        img_file_path = self.scene_info.get_img_file_path(img_name)
        img = cv.imread(img_file_path, None)
        if self.show_input_img:
            plt.figure(figsize=(9, 9))
            plt.title(img_name)
            plt.imshow(img)
            show_or_close(True)

        orig_height = img.shape[0]
        orig_width = img.shape[1]
        if self.estimate_k:
            focal_length = (orig_width + orig_height) * self.focal_point_mean_factor
            K_for_rectification = np.array([
                [focal_length, 0,            orig_width / 2.0],
                [0,            focal_length, orig_height / 2.0],
                [0,            0,            1]
            ])
            real_K = K_for_rectification
        else:
            real_K = self.scene_info.get_img_K(img_name)
            K_for_rectification = real_K
            focal_length = real_K[0, 0]
            assert abs(real_K[0, 2] * 2 - orig_width) < 0.5
            assert abs(real_K[1, 2] * 2 - orig_height) < 0.5

        if not self.rectify:
            kps, descs = self.feature_descriptor.detectAndCompute(img, None)

            Timer.end_check_point("processing img")
            return ImageData(img=img,
                             real_K=real_K,
                             key_points=kps,
                             descriptions=descs,
                             normals=None,
                             components_indices=None,
                             valid_components_dict=None)

        else:

            img_data_path = "{}/{}_img_data.pkl".format(img_processing_dir, img_name)
            if self.use_cached_img_data and os.path.isfile(img_data_path):
                Timer.start_check_point("reading img processing data")
                with open(img_data_path, "rb") as f:
                    print("img data for {} already computed, reading: {}".format(img_name, img_data_path))
                    img_serialized_data: ImageSerializedData = pickle.load(f)
                Timer.end_check_point("reading img processing data")
                return ImageData.from_serialized_data(img=img,
                                                      real_K=real_K,
                                                      img_serialized_data=img_serialized_data)

            Timer.start_check_point("processing img from scratch")

            depth_data_file_name = "{}.npy".format(img_name)
            normals, _ = compute_only_normals(focal_length,
                                                 orig_height,
                                                 orig_width,
                                                 self.depth_input_dir,
                                                 depth_data_file_name,
                                                 simple_weighing=True)

            filter_mask = get_nonsky_mask(img, normals.shape[0], normals.shape[1])

            sky_out_path = "{}/{}_sky_mask.jpg".format(img_processing_dir, img_name)
            show_sky_mask(img, filter_mask, img_name, show=self.show_sky_mask, save=self.save_sky_mask, path=sky_out_path)

            normals_clusters_repr, normal_indices = cluster_normals(normals, filter_mask=filter_mask)

            degrees_list = get_degrees_between_normals(normals_clusters_repr)
            if not self.stats["imgs_data"].__contains__(img_name):
                self.stats["imgs_data"][img_name] = {}
            self.stats["imgs_data"][img_name]["deg_between_normals"] = degrees_list

            show_or_save_clusters(normals,
                                  normal_indices,
                                  normals_clusters_repr,
                                  img_processing_dir,
                                  depth_data_file_name,
                                  show=self.show_clusters,
                                  save=self.save_clusters)


            if self.upsample_early:
                normal_indices = possibly_upsample_normals(img, normal_indices)

            valid_normal_indices = []
            for i, normal in enumerate(normals_clusters_repr):
                angle_rad = math.acos(np.dot(normal, np.array([0, 0, -1])))
                angle_degrees = angle_rad * 180 / math.pi
                # print("angle: {} vs. angle threshold: {}".format(angle_degrees, Config.plane_threshold_degrees))
                if angle_degrees >= Config.plane_threshold_degrees:
                    # print("WARNING: two sharp of an angle with the -z axis, skipping the rectification")
                    continue
                else:
                    # print("angle ok")
                    valid_normal_indices.append(i)

            components_indices, valid_components_dict = get_connected_components(normal_indices, valid_normal_indices,
                                                                                 closing_size=self.connected_components_closing_size,
                                                                                 flood_filling=self.connected_components_flood_fill,
                                                                                 connectivity=self.connected_components_connectivity)

            if not self.upsample_early:
                assert np.all(components_indices < 256), "could not retype to np.uint8"
                components_indices = components_indices.astype(dtype=np.uint8)
                components_indices = possibly_upsample_normals(img, components_indices)
                components_indices = components_indices.astype(dtype=np.uint32)

            components_out_path = "{}/{}_cluster_connected_components".format(img_processing_dir, img_name)
            get_and_show_components(components_indices,
                                    valid_components_dict,
                                    normals=normals_clusters_repr,
                                    show=self.show_clustered_components,
                                    save=self.save_clustered_components,
                                    path=components_out_path,
                                    file_name=depth_data_file_name[:-4])

            # get rectification
            rectification_path_prefix = "{}/{}".format(img_processing_dir, img_name)
            kps, descs = get_rectified_keypoints(normals_clusters_repr,
                                                 components_indices,
                                                 valid_components_dict,
                                                 img,
                                                 K_for_rectification,
                                                 descriptor=self.feature_descriptor,
                                                 img_name=img_name,
                                                 clip_angle=self.clip_angle,
                                                 show=self.show_rectification,
                                                 save=self.save_rectification,
                                                 out_prefix=rectification_path_prefix
                                                 )

            img_data = ImageData(img=img,
                                 real_K=real_K,
                                 key_points=kps,
                                 descriptions=descs,
                                 normals=normals_clusters_repr,
                                 components_indices=components_indices,
                                 valid_components_dict=valid_components_dict)

            Timer.end_check_point("processing img from scratch")

            Timer.start_check_point("saving img data")
            with open(img_data_path, "wb") as f:
                print("img data for {} saving into: {}".format(img_name, img_data_path))
                pickle.dump(img_data.to_serialized_data(), f)
            Timer.end_check_point("saving img data")

            Timer.end_check_point("processing img")
            return img_data

    def compute_img_normals(self, img, img_name):

        Timer.start_check_point("processing img", img_name)
        print("processing img {}".format(img_name))

        # get focal_length
        orig_height = img.shape[0]
        orig_width = img.shape[1]
        if self.estimate_k:
            focal_length = (orig_width + orig_height) * self.focal_point_mean_factor
        else:
            real_K = self.scene_info.get_img_K(img_name)
            focal_length = real_K[0, 0]
            assert abs(real_K[0, 2] * 2 - orig_width) < 0.5
            assert abs(real_K[1, 2] * 2 - orig_height) < 0.5

        depth_data_file_name = "{}.npy".format(img_name)
        depth_data = read_depth_data(depth_data_file_name, self.depth_input_dir)

        img_processing_dir = "{}/imgs".format(self.output_dir)
        sky_out_path = "{}/{}_sky_mask.jpg".format(img_processing_dir, img_name)
        filter_mask = get_nonsky_mask(img, depth_data.shape[2], depth_data.shape[3])
        show_sky_mask(img, filter_mask, img_name, show=self.show_sky_mask, save=self.save_sky_mask, path=sky_out_path)

        for sigma in [0.8, 1.2, 1.6]:

            Config.svd_weighted_sigma = sigma

            normals, s_values = compute_normals_from_svd(focal_length,
                                                         orig_height,
                                                         orig_width,
                                                         depth_data,
                                                         simple_weighing=True)

            for mean_shift in ["full", "mean", None]:

                for singular_value_quantil in [0.6, 0.8, 1.0]:

                    for angle_distance_threshold_degrees in [20, 25, 30, 35]:

                        ms_str = "ms_{}".format(mean_shift)
                        params_key = "{}_{}_{}_{}".format(ms_str, singular_value_quantil, angle_distance_threshold_degrees, sigma)

                        print("Params: s_value_hist_ratio: {}, angle_distance_threshold_degrees: {}, sigma: {}, mean shift step: {}".format(singular_value_quantil, angle_distance_threshold_degrees, sigma, ms_str))

                        Clustering.angle_distance_threshold_degrees = angle_distance_threshold_degrees
                        Clustering.recompute(math.sqrt(singular_value_quantil))

                        smallest_singular_values = s_values[:, :, 2]
                        smallest_singular_values = smallest_singular_values / depth_data[0, 0]

                        w, h = smallest_singular_values.shape[0], smallest_singular_values.shape[1]
                        smallest_singular_values = smallest_singular_values.reshape(w * h)
                        sorted, indices = torch.sort(smallest_singular_values)

                        mask = torch.zeros_like(smallest_singular_values, dtype=torch.bool)
                        mask[indices[:(int(indices.shape[0] * singular_value_quantil))]] = True
                        mask = mask.reshape(w, h).numpy()

                        if singular_value_quantil != 1.0:
                            show_sky_mask(img, mask, img_name, show=self.show_sky_mask, save=False, title="quantile mask")
                            show_sky_mask(img, filter_mask & mask, img_name, show=self.show_sky_mask, save=False, title="quantile and sky mask")

                        cp_key = "clustering_{}".format(params_key)
                        Timer.start_check_point(cp_key)
                        normals_clusters_repr, normal_indices = cluster_normals(normals, filter_mask=filter_mask & mask, mean_shift=mean_shift)
                        Timer.end_check_point(cp_key)

                        sums = np.array([np.sum(normal_indices == i) for i in range(len(normals_clusters_repr))])
                        indices = np.argsort(-sums)

                        # then delete the previous two lines - or just debug this
                        # for i in range(len(indices)):
                        #     assert i == indices[i]

                        normals_clusters_repr_sorted = normals_clusters_repr[indices]

                        degrees_list = get_degrees_between_normals(normals_clusters_repr_sorted)
                        if not self.stats.keys().__contains__("normals_degrees"):
                            self.stats["normals_degrees"] = {}

                        if not self.stats["normals_degrees"].__contains__(params_key):
                            self.stats["normals_degrees"][params_key] = {}
                        self.stats["normals_degrees"][params_key][img_name] = degrees_list

                        show_or_save_clusters(normals,
                                              normal_indices,
                                              normals_clusters_repr,
                                              img_processing_dir,
                                              depth_data_file_name,
                                              show=self.show_clusters,
                                              save=self.save_clusters)

    def run_sequential_pipeline(self):

        self.start()

        file_names, _ = self.scene_info.get_megadepth_file_names_and_dir(self.sequential_files_limit, self.chosen_depth_files)
        for depth_data_file_name in file_names:
            self.process_image(depth_data_file_name[:-4])

        self.save_stats("sequential")

    def show_and_read_img(self, img_name):
        img_file_path = self.scene_info.get_img_file_path(img_name)
        img = cv.imread(img_file_path, None)
        if self.show_input_img:
            plt.figure(figsize=(9, 9))
            plt.title(img_name)
            plt.imshow(img)
            show_or_close(True)
        return img

    def run(self):
        if self.method == "compute_normals":
            self.run_sequential_for_normals()
        elif self.method == "run_sequential_pipeline":
            self.run_sequential_pipeline()
        elif self.method == "run_matching_pipeline":
            self.run_matching_pipeline()
        else:
            print("incorrect value of '{}' for method. Choose one from 'compute_normals', 'run_sequential_pipeline' or 'run_matching_pipeline'".format(self.method))

    def run_sequential_for_normals(self):

        self.start()

        file_names, _ = self.scene_info.get_megadepth_file_names_and_dir(self.sequential_files_limit, self.chosen_depth_files)
        file_names_permuted = [file_names[two_hundred_permutation[i]] for i in range(self.permutation_limit)]
        for i, depth_data_file_name in enumerate(file_names_permuted):
            img_name = depth_data_file_name[:-4]
            img = self.show_and_read_img(img_name)
            self.compute_img_normals(img, img_name)
            if i % 10 == 0:
                self.save_stats("normals_{}".format(i))
            evaluate_normals(self.stats)
            Timer.end()

        self.save_stats("normals")

    def run_matching_pipeline(self):

        self.start()

        stats_map = {}
        already_processed = set()

        for difficulty in self.matching_difficulties:
            print("Difficulty: {}".format(difficulty))

            stats_map_diff = {}

            processed_pairs = 0
            for img_pair in self.scene_info.img_pairs_lists[difficulty]:

                key = SceneInfo.get_key(img_pair.img1, img_pair.img2)
                if self.matching_pairs is not None:
                    if key not in self.matching_pairs or key in already_processed:
                        continue
                    else:
                        already_processed.add(key)

                if self.matching_pairs is None and self.matching_limit is not None and processed_pairs >= self.matching_limit:
                    print("Reached matching limit of {} for difficulty {}".format(self.matching_limit, difficulty))
                    break

                Timer.start_check_point("complete image pair matching")
                print("Will be matching pair {}".format(key))

                matching_out_dir = "{}/matching".format(self.output_dir)
                Path(matching_out_dir).mkdir(parents=True, exist_ok=True)

                # I might not need normals yet
                # img1, K_1, kps1, descs1, normals1, components_indices1, valid_components_dict1
                try:
                    image_data1 = self.process_image(img_pair.img1)
                except Exception as e:
                    print("{} couldn't be processed, skipping the matching pair {}_{}".format(img_pair.img1,
                                                                                              img_pair.img1,
                                                                                              img_pair.img2))
                    print(traceback.format_exc(), file=sys.stderr)
                    continue

                try:
                    image_data2 = self.process_image(img_pair.img2)
                except Exception as e:
                    print("{} couldn't be processed, skipping the matching pair {}_{}".format(img_pair.img2,
                                                                                              img_pair.img1,
                                                                                              img_pair.img2))
                    print(traceback.format_exc(), file=sys.stderr)
                    continue

                if self.planes_based_matching:
                    E, inlier_mask, src_pts, dst_pts, tentative_matches = match_images_with_dominant_planes(
                        image_data1,
                        image_data2,
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
                        image_data1.img,
                        image_data1.key_points,
                        image_data1.descriptions,
                        image_data1.real_K,
                        image_data2.img,
                        image_data2.key_points,
                        image_data2.descriptions,
                        image_data2.real_K,
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
                        image_data1.img,
                        image_data1.key_points,
                        image_data1.descriptions,
                        image_data1.real_K,
                        image_data2.img,
                        image_data2.key_points,
                        image_data2.descriptions,
                        image_data2.real_K,
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

                evaluate_matching(self.scene_info,
                                  E,
                                  image_data1.key_points,
                                  image_data2.key_points,
                                  tentative_matches,
                                  inlier_mask,
                                  img_pair,
                                  stats_map_diff,
                                  image_data1.normals,
                                  image_data2.normals,
                                  )

                processed_pairs = processed_pairs + 1
                Timer.end_check_point("complete image pair matching")

            if len(stats_map_diff) > 0:
                stats_map[difficulty] = stats_map_diff
                stats_file_name = "{}/stats_diff_{}.pkl".format(self.output_dir, difficulty)
                with open(stats_file_name, "wb") as f:
                    pickle.dump(stats_map_diff, f)
                print("Stats for difficulty {}:".format(difficulty))
                print("Group\tAcc.(5º)")
                #evaluate_percentage_correct(stats_map_diff, difficulty, n_worst_examples=10, th_degrees=5)

        all_stats_file_name = "{}/all.stats.pkl".format(self.output_dir)
        with open(all_stats_file_name, "wb") as f:
            pickle.dump(stats_map, f)

        self.save_stats("matching")
        self.log()
        #evaluate_all(stats_map, n_worst_examples=None)

    def save_stats(self, key):
        file_name = "{}/stats_{}_{}.pkl".format(self.output_dir, key, get_tmsp())
        with open(file_name, "wb") as f:
            pickle.dump(self.stats, f)
            print("stats saved")


def get_tmsp():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S_%f")


def append_all(pipeline, str):
    use_degensac = "DEGENSAC" if pipeline.use_degensac else "RANSAC"
    estimate_K = "estimatedK" if pipeline.estimate_k else "GTK"
    rectified = "rectified" if pipeline.rectify else "unrectified"
    timestamp = get_tmsp()
    return "{}_{}_{}_{}_{}_{}_{}".format(str, pipeline.scene_type, pipeline.scene_name.replace("/", "_"), use_degensac, estimate_K, rectified, timestamp)


def main():

    parser = argparse.ArgumentParser(prog='pipeline')
    parser.add_argument('--output_dir', help='output dir')
    args = parser.parse_args()

    Timer.start()

    pipeline = Pipeline.configure("config.txt", args)
    pipeline.run()

    Timer.end()


if __name__ == "__main__":
    main()
