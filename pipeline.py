
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import cv2 as cv
import pickle
import traceback
import sys

import torch
import argparse

from config import Config
from connected_components import get_connected_components, get_and_show_components
from depth_to_normals import compute_normals, compute_only_normals
from depth_to_normals import show_sky_mask, cluster_normals, show_or_save_clusters
from matching import match_find_E, match_find_F_degensac, match_images_with_dominant_planes
from rectification import possibly_upsample_normals, get_rectified_keypoints
from scene_info import SceneInfo
from utils import Timer
from img_utils import show_or_close
from evaluation import *
from sky_filter import get_nonsky_mask

import matplotlib.pyplot as plt


def parse_list(list_str: str):
    fields = list_str.split(",")
    fields = filter(lambda x: x != "", map(lambda x: x.strip(), fields))
    fields = list(fields)
    return fields


@dataclass
class Pipeline:

    scene_name = None
    scene_type = None
    file_name_suffix = None
    output_dir = None
    output_dir_prefix = None

    # actually unused
    show_save_normals = False
    show_orig_image = True

    chosen_depth_files = None
    sequential_files_limit = None

    show_input_img = True

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

    knn_ratio_threshold = 0.85

    use_cached_img_data = True

    @staticmethod
    def configure(config_file_name: str, args):

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
                elif k == "image_pairs":
                    pipeline.matching_pairs = parse_list(v)
                elif k == "chosen_depth_files":
                    pipeline.chosen_depth_files = parse_list(v)
                elif k == "use_cached_img_data":
                    pipeline.use_cached_img_data = v.lower() == "true"
                elif k == "output_dir_prefix":
                    pipeline.output_dir_prefix = v
                else:
                    print("WARNING - unrecognized param: {}".format(k))

        pipeline.matching_difficulties = list(range(matching_difficulties_min, matching_difficulties_max))

        if args.__contains__("output_dir"):
            pipeline.output_dir = args.output_dir
        else:
            pipeline.output_dir = append_all(pipeline, pipeline.output_dir_prefix)

        return pipeline

    def start(self):
        print("is torch.cuda.is_available(): {}".format(torch.cuda.is_available()))

        if self.matching_pairs is not None:
            self.matching_difficulties = range(0, 18)

        self.log()
        self.scene_info = SceneInfo.read_scene(self.scene_name, self.scene_type, file_name_suffix=self.file_name_suffix)
        self.depth_input_dir = self.scene_info.depth_input_dir()

    def log(self):
        print("Pipeline config:")
        attr_list = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for attr_name in attr_list:
            if attr_name in ["scene_info"]:
                continue
            print("  {} = {}".format(attr_name, getattr(self, attr_name)))
        print()

        Config.log()

    def process_image(self, img_name):

        Timer.start_check_point("processing img")
        print("Processing: {}".format(img_name))
        img_processing_dir = "{}/imgs".format(self.output_dir)
        Path(img_processing_dir).mkdir(parents=True, exist_ok=True)

        img_file_path = self.scene_info.get_img_file_path(img_name)
        img = cv.imread(img_file_path, None)
        if self.show_input_img:
            plt.figure()
            plt.title(img_name)
            plt.imshow(img)
            plt.show(block=False)

        orig_height = img.shape[0]
        orig_width = img.shape[1]
        if self.estimate_k:
            focal_length = (orig_width + orig_height) * self.focal_point_mean_factor
            K = np.array([
                [focal_length, 0,            orig_width / 2.0],
                [0,            focal_length, orig_height / 2.0],
                [0,            0,            1]
            ])
        else:
            K = self.scene_info.get_img_K(img_name)
            focal_length = K[0, 0]
            assert abs(K[0, 2] * 2 - orig_width) < 0.5
            assert abs(K[1, 2] * 2 - orig_height) < 0.5

        if not self.rectify:
            kps, descs = self.feature_descriptor.detectAndCompute(img, None)

            Timer.end_check_point("processing img")
            return ImageData(img=img,
                             K=K,
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
                return ImageData.from_serialized_data(img, K, img_serialized_data)

            Timer.start_check_point("processing img from scratch")

            depth_data_file_name = "{}.npy".format(img_name)
            normals = compute_only_normals(focal_length, orig_height, orig_width, self.depth_input_dir, depth_data_file_name)
            filter_mask = get_nonsky_mask(img, normals.shape[0], normals.shape[1])

            sky_out_path = "{}/{}_sky_mask.jpg".format(img_processing_dir, img_name[:-4])
            show_sky_mask(img, filter_mask, img_name, show=self.show_sky_mask, save=self.save_sky_mask, path=sky_out_path)

            normals_clusters_repr, normal_indices = cluster_normals(normals, filter_mask=filter_mask)

            show_or_save_clusters(normals,
                                  normal_indices,
                                  normals_clusters_repr,
                                  img_processing_dir,
                                  depth_data_file_name,
                                  show=self.show_clusters,
                                  save=self.save_clusters)

            # normal indices => cluster indices (maybe safe here?)
            # TODO after the call to get_connected_components?
            normal_indices = possibly_upsample_normals(img, normal_indices)

            valid_normal_indices = []
            for i, normal in enumerate(normals_clusters_repr):
                angle_rad = math.acos(np.dot(normal, np.array([0, 0, -1])))
                angle_degrees = angle_rad * 180 / math.pi
                #print("angle: {} vs. angle threshold: {}".format(angle_degrees, Config.plane_threshold_degrees))
                if angle_degrees >= Config.plane_threshold_degrees:
                #print("WARNING: two sharp of an angle with the -z axis, skipping the rectification")
                    continue
                else:
                    #print("angle ok")
                    valid_normal_indices.append(i)

            components_indices, valid_components_dict = get_connected_components(normal_indices, valid_normal_indices)

            components_out_path = "{}/{}_cluster_connected_components.jpg".format(img_processing_dir, img_name[:-4])
            get_and_show_components(components_indices,
                                    valid_components_dict,
                                    normals=normals_clusters_repr,
                                    show=self.show_clustered_components,
                                    save=self.save_clustered_components,
                                    path=components_out_path,
                                    file_name=depth_data_file_name[:-4])


            # matching_out_dir = "{}/matching".format(self.output_dir)
            # Path(matching_out_dir).mkdir(parents=True, exist_ok=True)

            # get rectification
            rectification_path_prefix = "{}/{}".format(img_processing_dir, img_name[:-4])
            kps, descs = get_rectified_keypoints(normals_clusters_repr,
                                                 components_indices,
                                                 valid_components_dict,
                                                 img,
                                                 K,
                                                 descriptor=self.feature_descriptor,
                                                 img_name=img_name,
                                                 show=self.show_rectification,
                                                 save=self.save_rectification,
                                                 out_prefix=rectification_path_prefix
                                                 )

            img_data = ImageData(img=img, K=K, key_points=kps, descriptions=descs, normals=normals_clusters_repr, components_indices=components_indices, valid_components_dict=valid_components_dict)

            Timer.end_check_point("processing img from scratch")

            Timer.start_check_point("saving img data")
            with open(img_data_path, "wb") as f:
                print("img data for {} saving into: {}".format(img_name, img_data_path))
                pickle.dump(img_data.to_serialized_data(), f)
            Timer.end_check_point("saving img data")

            Timer.end_check_point("processing img")
            return img_data

    def run_sequential_pipeline(self):

        self.start()

        file_names, _ = self.scene_info.get_megadepth_file_names_and_dir(self.sequential_files_limit, self.chosen_depth_files)
        for depth_data_file_name in file_names:
            self.process_image(depth_data_file_name[:-4])

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
                    # E, inlier_mask, src_pts, dst_pts, kps1, kps2, tentative_matches =
                    match_images_with_dominant_planes(
                        image_data1,
                        image_data2,
                        images_info=self.scene_info.img_info_map,
                        img_pair=img_pair,
                        out_dir=matching_out_dir,
                        show=self.show_matching,
                        save=self.save_matching)

                elif self.use_degensac:
                    E, inlier_mask, src_pts, dst_pts, tentative_matches = match_find_F_degensac(
                        image_data1.img,
                        image_data1.key_points,
                        image_data1.descriptions,
                        image_data1.K,
                        image_data2.img,
                        image_data2.key_points,
                        image_data2.descriptions,
                        image_data2.K,
                        img_pair,
                        matching_out_dir,
                        show=self.show_matching,
                        save=self.save_matching,
                        ratio_thresh=self.knn_ratio_threshold)

                else:
                    E, inlier_mask, src_pts, dst_pts, tentative_matches = match_find_E(
                        image_data1.img,
                        image_data1.key_points,
                        image_data1.descriptions,
                        image_data1.K,
                        image_data2.img,
                        image_data2.key_points,
                        image_data2.descriptions,
                        image_data2.K,
                        img_pair,
                        matching_out_dir,
                        show=self.show_matching,
                        save=self.save_matching,
                        ratio_thresh=self.knn_ratio_threshold)

                evaluate_matching(self.scene_info,
                                  E,
                                  image_data1.key_points,
                                  image_data2.key_points,
                                  tentative_matches,
                                  inlier_mask,
                                  img_pair,
                                  matching_out_dir,
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
                evaluate_percentage_correct(stats_map_diff, difficulty, n_worst_examples=10)

        all_stats_file_name = "{}/all.stats.pkl".format(self.output_dir)
        with open(all_stats_file_name, "wb") as f:
            pickle.dump(stats_map, f)

        evaluate_all(stats_map, n_worst_examples=None)


def append_all(pipeline, str):
    use_degensac = "DEGENSAC" if pipeline.use_degensac else "RANSAC"
    estimate_K = "estimatedK" if pipeline.estimate_k else "GTK"
    rectified = "rectified" if pipeline.rectify else "unrectified"
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    return "{}_{}_{}_{}_{}_{}_{}".format(str, pipeline.scene_type, pipeline.scene_name.replace("/", "_"), use_degensac, estimate_K, rectified, timestamp)


def main():

    parser = argparse.ArgumentParser(prog='pipeline')
    parser.add_argument('--output_dir', help='ouput dir')
    args = parser.parse_args()

    # hurts
    # "frame_0000001650_1_frame_0000000730_3"
    # "frame_0000000200_4_frame_0000001845_1"
    # helps
    # "frame_0000000730_4_frame_0000000390_4"

    Timer.start()

    Config.config_map[Config.key_planes_based_matching_merge_components] = False

    pipeline = Pipeline.configure("config.txt", args)

    #
    # pipeline.matching_pairs = [
    #     "5fd2c91236049babb753c6be7d99cdc_cb706ca5ce5c4625b7b29f1cced2418f",
    #     "e153533ee59843e0bce5072cd9c13a35_423d856c619a42a3aa341e41972bcc18",
    #     "f34c59d050448395101ac297d2fcf6_29c203e30ddd461cab3ed7a0225b0171",
    #     "d023d86287e04da0a7239b93e7dd260c_b65d3668d39e46ca9330a2abeb7c5285",
    #     "cb706ca5ce5c4625b7b29f1cced2418f_4a8e1cbfb45a45fc9e872b0e6b564741",
    #     "a4a010b9ff7143b0aaee9f53e000b3e6_7ce20ade05044dffb0220dd2dcb31628",
    #     "a3c549a16b4981bb400614da79ccac_0a2cccf4559a422bbb3f59c555793f31",
    #     "d36717005c624a678ac1cfa9eb5a5190_c92027d4a43e45d0822e4fd932b9d1f1",
    #     "d1f0979446ea68291618a2dc175_0a2cccf4559a422bbb3f59c555793f31",
    #     "c92027d4a43e45d0822e4fd932b9d1f1_57050d1f0979446ea68291618a2dc175",
    # ]

    # pipeline.matching_pairs = [
    #     "b0695cf27e3a45699cabafa926b6ad3f_2c18960517b24892a5ffb52341ece9eb"
    #     "8d377076df88437daccbf07cdf4f3e5a_0b1ea4bbfbeb45428631265fb85ab89b",
    #     "48767b964d9b49078e4015d02c98ce76_01bbad62912c4bde8586a16786c35db5",
    #     "f5fd2c91236049babb753c6be7d99cdc_01522fc97a924bf1adec026152c6305f",
    #     "cdd3fd3f2b4948438b715e1f7963cf77_9c453410e2ed4bd0bd288d7bf2308fe3",
    #     "2c18960517b24892a5ffb52341ece9eb_2b5315968bc5468c995b978620879439",
    #     "fe9eff017cbc45dca2ad0a7037b16e6b_d36717005c624a678ac1cfa9eb5a5190",
    #     "ec28d890df844db48a3d20f5bd8ea80b_01522fc97a924bf1adec026152c6305f",
    #     "9ccf06e9e3fe42f4933c09a55e632395_043f6f1c4dba43a5b99a81ff3b8780e8",
    #     "e4c503ff3448479db49f90d833a8e2b6_043f6f1c4dba43a5b99a81ff3b8780e8",
    # ]
    #pipeline.rectify = True

    pipeline.run_matching_pipeline()

    # pipeline.rectify = False
    # pipeline.run_matching_pipeline()

    Timer.end()

    # RECT
    # frame_0000000750_1_frame_0000001460_3 : 0.2975730073440412 : 0
    # frame_0000001280_2_frame_0000000435_1 : 0.3117467900844886 : 0
    # frame_0000000045_1_frame_0000001465_4 : 0.3475394532917147 : 0
    # frame_0000001670_1_frame_0000000705_3 : 0.35030656173324887 : 0
    # frame_0000000695_3_frame_0000000535_4 : 0.38110056845168916 : 0
    # frame_0000001155_1_frame_0000001330_1 : 0.396294848576985 : 0
    # frame_0000001650_1_frame_0000000730_3 : 0.41452517328156235 : 0
    # frame_0000000045_2_frame_0000002230_1 : 0.8421649629297145 : 0
    # frame_0000000045_1_frame_0000001460_4 : 1.2111855989905465 : 0
    # frame_0000001535_4_frame_0000000305_1 : 1.6491643182932554 : 0
    # frame_0000001625_4_frame_0000001520_4 : 3.1229108422181784 : 0

    # NO RECT
    # frame_0000000420_2_frame_0000000755_3 : 0.2395142577686067 : 0
    # frame_0000001935_1_frame_0000000640_3 : 0.2710834091883783 : 0
    # frame_0000000770_4_frame_0000000685_4 : 0.2774820857254884 : 0
    # frame_0000001480_3_frame_0000001190_2 : 0.3206430790811858 : 0
    # frame_0000001145_2_frame_0000001430_3 : 0.32928574716602327 : 0
    # frame_0000000660_3_frame_0000001890_1 : 0.3412612633633496 : 0
    # frame_0000001650_1_frame_0000000730_3 : 0.4171382008287733 : 0
    #

    #pipeline.matching_pairs = "frame_0000000650_2_frame_0000001285_2"
    # pipeline.matching_pairs = ["frame_0000000750_1_frame_0000001460_3",
    # "frame_0000001280_2_frame_0000000435_1",
    # "frame_0000000045_1_frame_0000001465_4",
    # "frame_0000001670_1_frame_0000000705_3",
    # "frame_0000000695_3_frame_0000000535_4",
    # "frame_0000001155_1_frame_0000001330_1",
    # "frame_0000001650_1_frame_0000000730_3",
    # "frame_0000000045_2_frame_0000002230_1",
    # "frame_0000000045_1_frame_0000001460_4",
    # "frame_0000001535_4_frame_0000000305_1",
    # "frame_0000001625_4_frame_0000001520_4"]

    # pipeline.matching_pairs = "frame_0000000045_1_frame_0000001465_4"

    # pipeline.matching_pairs = [
    #     "frame_0000000045_1_frame_0000001465_4",
    #     "frame_0000000045_1_frame_0000001460_4",
    #     "frame_0000000675_1_frame_0000000045_1",
    #     ]

    # relict hurts
    # frame_0000000175_1_frame_0000000845_1: 3.134319304222711
    # frame_0000000200_4_frame_0000001845_1: 3.1340711124831624
    # frame_0000001350_2_frame_0000001015_1: 3.0998686019217874
    # frame_0000001075_3_frame_0000001365_4: 3.079703309390179
    # frame_0000001720_1_frame_0000000030_2: 3.0589784436231784
    # frame_0000001740_3_frame_0000001020_1: 3.044786897719698
    # frame_0000000640_4_frame_0000000055_4: 3.0038974081353076
    # frame_0000001590_4_frame_0000000840_1: 2.994597958434275
    # frame_0000000990_1_frame_0000001565_4: 2.9864855661578193
    # frame_0000000585_2_frame_0000002160_1: 2.9804281328592155


    #pipeline.chosen_depth_files = ["frame_0000001465_4.npy"]
    # pipeline.chosen_depth_files = ["frame_0000000045_1.npy"]
    #
    # pipeline.matching_limit = 100
    # pipeline.rectify = True

    #pipeline.run_sequential_pipeline()

    # TODO good example!!!
    #"frame_0000000675_1_frame_0000000045_1",


if __name__ == "__main__":
    main()
