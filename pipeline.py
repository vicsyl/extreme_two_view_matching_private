from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import cv2 as cv
import pickle
import traceback
import sys

from config import Config
from connected_components import get_connected_components, show_components
from depth_to_normals import compute_normals, get_megadepth_file_names_and_dir, megadepth_input_dir
from image_data import ImageData
from matching import match_images_and_keypoints, match_images_with_dominant_planes
from rectification import possibly_upsample_normals, get_rectified_keypoints
from scene_info import SceneInfo
from utils import Timer
from evaluation import *


@dataclass
class Pipeline:

    scene_name = None
    output_dir = None

    show_save_normals = False

    chosen_depth_files = None
    sequential_files_limit = None

    show_clustered_components = False
    show_rectification = False

    #matching
    feature_descriptor = None
    matching_dir = None
    matching_difficulties = None
    matching_limit = None

    planes_based_matching = False

    rectify = True

    @staticmethod
    def read_conf(config_file_name: str):

        feature_descriptors_str_map = {
            "SIFT": cv.SIFT_create(),
        }

        pipeline = Pipeline()

        with open(config_file_name) as f:
            for line in f:

                if line.startswith("#"):
                    continue

                k, v = line.partition("=")[::2]
                k = k.strip()
                v = v.strip()

                if k == "scene_name":
                    pipeline.scene_name = v
                if k == "rectify":
                    pipeline.rectify = v.lower() == "true"
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
                elif k == "output_dir_prefix":
                    pipeline.output_dir = append_timestamp(v)
                elif k == "show_save_normals":
                    pipeline.show_save_normals = v.lower() == "true"
                elif k == "do_flann":
                    Config.config_map[Config.do_flann()] = v.lower() == "true"

        pipeline.matching_difficulties = list(range(matching_difficulties_min, matching_difficulties_max))

        return pipeline

    def start(self):
        self.log()
        self.scene_info = SceneInfo.read_scene(self.scene_name, lazy=False)
        self.depth_input_dir = megadepth_input_dir(self.scene_name)
        Config.set_rectify(self.rectify)
        Config.config_map[Config.save_normals_in_img] = self.show_save_normals
        Config.config_map[Config.show_normals_in_img] = self.show_save_normals

    def log(self):
        print("Pipeline config:")
        attr_list = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for attr_name in attr_list:
            print("  {} = {}".format(attr_name, getattr(self, attr_name)))
        print()

        Config.log()

    def process_image(self, img_name):

        Timer.start_check_point("processing img")
        print("Processing: {}".format(img_name))
        # TODO skip/override existing (on multiple levels)

        # input image & K
        img_file_path = self.scene_info.get_img_file_path(img_name)
        img = cv.imread(img_file_path, None)
        K = self.scene_info.get_img_K(img_name)

        # depth => indices
        normals_output_directory = "{}/normals/{}".format(self.output_dir, img_name)
        depth_data_file_name = "{}.npy".format(img_name)
        normals, normal_indices = compute_normals(self.scene_info, self.depth_input_dir, depth_data_file_name, normals_output_directory)
        # TODO - shouldn't the normals be persisted already with the connected components?

        # normal indices => cluster indices (maybe safe here?)
        normal_indices = possibly_upsample_normals(img, normal_indices)
        components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals)))
        if self.show_clustered_components:
            show_components(components_indices, valid_components_dict, normals=normals)

        # TODO if False, I can skip computing the normals !!!
        if Config.rectify():

            # get rectification
            kps, descs = get_rectified_keypoints(normals,
                                                 components_indices,
                                                 valid_components_dict,
                                                 img,
                                                 K,
                                                 descriptor=self.feature_descriptor,
                                                 img_name=img_name,
                                                 show=self.show_rectification)

        else:
            kps, descs = self.feature_descriptor.detectAndCompute(img, None)

        Timer.end_check_point("processing img")
        return ImageData(img=img, K=K, key_points=kps, descriptions=descs, normals=normals, components_indices=components_indices, valid_components_dict=valid_components_dict)

    def run_sequential_pipeline(self):

        self.start()

        file_names, _ = get_megadepth_file_names_and_dir(self.scene_name, self.sequential_files_limit, self.chosen_depth_files)
        for depth_data_file_name in file_names:
            self.process_image(depth_data_file_name[:-4])

    def run_matching_pipeline(self):

        self.start()

        stats_map = {}

        for difficulty in self.matching_difficulties:
            print("Difficulty: {}".format(difficulty))

            processed_pairs = 0
            for img_pair in self.scene_info.img_pairs_lists[difficulty]:
                if self.matching_limit is not None and processed_pairs >= self.matching_limit:
                    break

                Timer.start_check_point("complete image pair matching")

                matching_out_dir = "{}/matching".format(self.output_dir)

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

                Path(matching_out_dir).mkdir(parents=True, exist_ok=True)

                if self.planes_based_matching:
                    # E, inlier_mask, src_pts, dst_pts, kps1, kps2, tentative_matches =
                    match_images_with_dominant_planes(
                        image_data1,
                        image_data2,
                        images_info=self.scene_info.img_info_map,
                        img_pair=img_pair,
                        out_dir=matching_out_dir,
                        show=True,
                        save=True)

                else:
                    E, inlier_mask, src_pts, dst_pts, tentative_matches = match_images_and_keypoints(
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
                        show=True,
                        save=True)

                evaluate_matching(self.scene_info,
                                  E,
                                  image_data1.key_points,
                                  image_data2.key_points,
                                  tentative_matches,
                                  inlier_mask,
                                  img_pair,
                                  matching_out_dir,
                                  stats_map)

                processed_pairs = processed_pairs + 1
                Timer.end_check_point("complete image pair matching")

        all_stats_file_name = "{}/all.stats.pkl".format(self.output_dir)
        with open(all_stats_file_name, "wb") as f:
            pickle.dump(stats_map, f)

        evaluate(stats_map, self.scene_info)


def append_timestamp(str):
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
    return "{}_{}".format(str, timestamp)


def main():

    Timer.start()

    Config.set_rectify(False)
    Config.config_map[Config.key_planes_based_matching_merge_components] = False

    pipeline = Pipeline.read_conf("config.txt")

    pipeline.run_matching_pipeline()

    Timer.end()


if __name__ == "__main__":
    main()
