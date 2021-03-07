from scene_info import SceneInfo
from depth_to_normals import compute_normals_simple_diff_convolution, get_megadepth_file_names_and_dir
from dataclasses import dataclass
from rectification import possibly_upsample_normals, get_rectified_keypoints
from connected_components import get_connected_components, show_components
from utils import Timer

import cv2 as cv


@dataclass
class Pipeline:

    scene_name: str

    depth_files_limit: int
    chosen_depth_files: int

    save_normals: bool
    normals_dir: str

    def __post_init__(self):
        self.scene_info = SceneInfo.read_scene(self.scene_name)
        self.show_clustered_components = True

    def run_pipeline(self):

        Timer.start()

        self.scene_info = SceneInfo.read_scene(self.scene_name)
        file_names, depth_data_input_directory = get_megadepth_file_names_and_dir(self.scene_name, self.depth_files_limit, self.chosen_depth_files)

        for depth_data_file_name in file_names:

            img_name = depth_data_file_name[:-4]
            print("Processing: {}".format(depth_data_file_name))
            # TODO skip existing?

            # depth => indices
            output_directory = "{}/{}".format(self.normals_dir, img_name)
            normals, normal_indices = compute_normals_simple_diff_convolution(self.scene_info, depth_data_input_directory,
                                                                              depth_data_file_name, self.save_normals,
                                                                              output_directory)

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
            get_rectified_keypoints(normals, components_indices, valid_components_dict, img, K, descriptor=cv.SIFT_create(), img_name=img_name)

        Timer.check_point("Done processing {} imgs".format(len(file_names)))


def main():
    pipeline = Pipeline(scene_name="scene1",
                         depth_files_limit=20,
                         chosen_depth_files=None,
                         save_normals=False,
                         normals_dir="work/scene1/normals/simple_diff_mask")
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
