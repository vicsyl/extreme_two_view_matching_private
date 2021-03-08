from scene_info import SceneInfo
from depth_to_normals import compute_normals_simple_diff_convolution, get_megadepth_file_names_and_dir, megadepth_input_dir
from dataclasses import dataclass
from rectification import possibly_upsample_normals, get_rectified_keypoints
from connected_components import get_connected_components, show_components
from utils import Timer
from pathlib import Path
from matching import match_images_and_keypoints

import cv2 as cv


@dataclass
class Pipeline:

    scene_name: str

    sequential_files_limit: int
    chosen_depth_files: int

    save_normals: bool
    normals_dir: str

    #matching
    feature_descriptor: cv.Feature2D
    matching_dir: str
    matching_difficulties: list
    matching_limit: str

    def __post_init__(self):
        Timer.start()
        self.scene_info = SceneInfo.read_scene(self.scene_name)
        Timer.check_point("scene info read")
        self.show_clustered_components = True
        self.depth_input_dir = megadepth_input_dir(self.scene_name)

    def process_file(self, img_name):

        print("Processing: {}".format(img_name))
        # TODO skip/override existing?

        # depth => indices
        output_directory = "{}/{}".format(self.normals_dir, img_name)
        normals, normal_indices = compute_normals_simple_diff_convolution(self.scene_info, self.depth_input_dir, "{}.npy".format(img_name), self.save_normals, output_directory)

        # TODO - shouldn't the normals be persisted already with the connected components?

        # read the input image
        img_file_path = self.scene_info.get_img_file_path(img_name)
        img = cv.imread(img_file_path, None)

        # normal indices => cluster indices (maybe safe here?)
        normal_indices = possibly_upsample_normals(img, normal_indices)
        components_indices, valid_components_dict = get_connected_components(normal_indices, range(len(normals)), True)
        if self.show_clustered_components:
            show_components(components_indices, valid_components_dict.keys())

        # get rectification
        K = self.scene_info.get_img_K(img_name)
        kps, descs = get_rectified_keypoints(normals, components_indices, valid_components_dict, img, K, descriptor=self.feature_descriptor, img_name=img_name)
        return img, K, kps, descs

    def run_sequential_pipeline(self):

        Timer.check_point("Running sequential pipeline")

        file_names, _ = get_megadepth_file_names_and_dir(self.scene_name, self.sequential_files_limit, self.chosen_depth_files)
        for depth_data_file_name in file_names:
            self.process_file(depth_data_file_name[:-4])

        Timer.check_point("Done processing {} imgs".format(len(file_names)))

    def run_matching_pipeline(self):

        Timer.check_point("Running matching pipeline")

        processed_pairs = 0
        for difficulty in self.matching_difficulties:

            print("Difficulty: {}".format(difficulty))

            for img_pair in self.scene_info.img_pairs[difficulty]:

                if self.matching_limit is not None and processed_pairs >= self.matching_limit:
                    break

                out_dir = "work/{}/matching/{}/{}_{}".format(self.scene_info.name, self.matching_dir, img_pair.img1, img_pair.img2)

                img1, K_1, kps1, descs1 = self.process_file(img_pair.img1)
                if img1 is None:
                    print("{} couldn't be processed, skipping the matching pair {}_{}".format(img_pair.img1, img_pair.img1, img_pair.img2))
                    continue

                img2, K_2, kps2, descs2 = self.process_file(img_pair.img2)
                if img2 is None:
                    print("{} couldn't be processed, skipping the matching pair {}_{}".format(img_pair.img2, img_pair.img1, img_pair.img2))
                    continue

                Path(out_dir).mkdir(parents=True, exist_ok=True)

                E, inlier_mask, src_pts, dst_pts, kps1, kps2 = match_images_and_keypoints(img1, kps1, descs1, K_1, img2, kps2, descs2, K_2, self.scene_info.img_info_map, img_pair, out_dir, show=True, save=True)
                processed_pairs = processed_pairs + 1

        Timer.check_point("Done processing {} image pairs".format(processed_pairs))


def main():
    pipeline = Pipeline(scene_name="scene1",
                        sequential_files_limit=10,
                        chosen_depth_files=None,
                        save_normals=False,
                        matching_dir="pipeline_with_rectification",
                        matching_difficulties=[0],
                        matching_limit=2,
                        feature_descriptor=cv.SIFT_create(),
                        normals_dir="work/scene1/normals/simple_diff_mask")

    #pipeline.run_sequential_pipeline()
    pipeline.run_matching_pipeline()

    matching_difficulties: list
    matching_limit: str


if __name__ == "__main__":
    main()
