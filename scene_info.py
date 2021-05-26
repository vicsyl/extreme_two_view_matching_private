import numpy as np
import glob
import h5py
import os

from dataclasses import dataclass
from utils import Timer, quaternions_to_R


def read_google_scene(scene_name):

    paths_h5 = glob.glob("googleurban/{}/set_100/calibration/*.h5".format(scene_name))
    f = h5py.File(paths_h5[0], "r")
    print(f.keys())
    for key in f.keys():
        print("{}: {}".format(key, f[key][()]))

    img_pairs_lists = {}
    img_pairs_maps = {}
    image_info_map = {}

    for diff in range(10):

        img_pairs_lists[diff] = []
        img_pairs_maps[diff] = {}

        print("Diff: {}".format(diff))
        file_name = "googleurban/{}/set_100/new-vis-pairs/keys-th-0.{}.npy".format(scene_name, diff)
        data_np = np.load(file_name)

        counter = 0
        for i in range(data_np.shape[0]):
            pair = data_np[i].split("-")
            img1_exists = os.path.isfile("googleurban/{}/set_100/images/{}.png".format(scene_name, pair[0]))
            img2_exists = os.path.isfile("googleurban/{}/set_100/images/{}.png".format(scene_name, pair[1]))
            cal1_file = "googleurban/{}/set_100/calibration/calibration_{}.h5".format(scene_name, pair[0])
            cal1_exists = os.path.isfile(cal1_file)
            cal2_file = "googleurban/{}/set_100/calibration/calibration_{}.h5".format(scene_name, pair[1])
            cal2_exists = os.path.isfile(cal2_file)
            if img1_exists and img2_exists and cal1_exists and cal2_exists:
                counter = counter + 1

                entry = ImagePairEntry(pair[0], pair[1], diff)
                img_pairs_maps[diff][data_np[i]] = entry
                img_pairs_lists[diff].append(entry)

                if not image_info_map.__contains__(pair[0]):
                    h5_data = h5py.File(cal1_file, "r")
                    K = h5_data["K"][()]
                    R = h5_data["R"][()]
                    T = h5_data["T"][()]
                    q = h5_data["q"][()]
                    cal1_file
                    image_info_map[pair[0]] = ImageEntry(pair[0], image_id=None, camera_id=None, qs=q, t=T, R=R, K=K)

        print("{} valid pair for diff {}".format(counter, diff))
    #
    # img_pair = ImagePairEntry(img1, img2, diff)
    #
    # img_pairs_lists[img_pair.difficulty].append(img_pair)
    # key = "{}_{}".format(img_pair.img1, img_pair.img2)
    # img_pairs_maps[img_pair.difficulty][key] = img_pair

    #image_info_map[name] = ImageEntry(name, image_id, camera_id, qs, ts)

    return SceneInfo(img_pairs_lists, img_pairs_maps, image_info_map, cameras=None, name=scene_name, type="google")


"""
Classes to read info about the data sets (info about matching pairs of images, cameras and points in the images)
"""

@dataclass
class ImagePairEntry:
    img1: str
    img2: str
    difficulty: int


@dataclass
class ImageEntry:
    image_name: str
    image_id: int
    camera_id: int

    qs: (float, float, float, float)
    t: (float, float, float)

    # let's check that or qs
    R: np.ndarray
    K: np.ndarray

    def __post_init__(self):
        if self.R is None:
            assert self.qs is not None
            self.R = quaternions_to_R(self.qs)

    def read_data_from_line(self, line):
        data = np.fromstring(line.strip(), dtype=np.float32, sep=" ")
        data = data.reshape((data.shape[0] // 3, 3))
        data_indices = data[:, 2].astype(dtype=np.int32).reshape(data.shape[0])
        data = data[:, :2]
        self.data = data
        self.data_point_idxs = data_indices
        # self.data = data[data_indices != -1]
        # self.ids = data_indices[data_indices != -1]


@dataclass
class CameraEntry:

    id: int
    model: str
    height_width: (int, int)
    focal_length: float
    # TODO it's never used as a tuple.... ( -> principal_point_x, principal_point_y)
    principal_point_x_y: (int, int)
    distortion: float

    def height(self):
        return self.height_width[0]

    def width(self):
        return self.height_width[1]

    def get_K(self):
        K = np.array([
            [self.focal_length,                 0, self.principal_point_x_y[0]],
            [                0, self.focal_length, self.principal_point_x_y[1]],
            [                0,                 0,                          1]
        ])
        return K


@dataclass
class SceneInfo:

    img_pairs_lists: list
    img_pairs_maps: list
    img_info_map: dict
    cameras: dict
    name: str
    type: str # "orig", "google"

    def get_input_dir(self):
        if self.type == "orig":
            return "original_dataset/{}/images".format(self.name)
        elif self.type == "google":
            return "googleurban/{}/set_100/images".format(self.name)
        else:
            raise Exception("unexpected type: {}".format(self.type))

    def get_img_file_path(self, img_name):
        if self.type == "orig":
            return '{}/{}.jpg'.format(self.get_input_dir(), img_name)
        elif self.type == "google":
            return '{}/{}.png'.format(self.get_input_dir(), img_name)
        else:
            raise Exception("unexpected type: {}".format(self.type))

    def get_img_K(self, img_name):
        img = self.img_info_map[img_name]
        if img.K is not None:
            return img.K
        else:
            return self.cameras[img.camera_id].get_K()

    def depth_input_dir(self):
        if self.type == "orig":
            return "depth_data/mega_depth/{}".format(self.name)
        elif self.type == "google":
            return "depth_data/googleurban/{}".format(self.name)
        else:
            raise Exception("unexpected type: {}".format(self.type))

    def get_megadepth_file_names_and_dir(self, limit, interesting_files):
        directory = self.depth_input_dir()
        file_names = SceneInfo.get_file_names_from_dir(directory, limit, interesting_files, ".npy")
        return file_names, directory

    @staticmethod
    def get_file_names_from_dir(input_dir: str, limit: int, interesting_files: list, suffix: str):
        if interesting_files is not None:
            return interesting_files
        else:
            return SceneInfo.get_file_names(input_dir, suffix, limit)

    @staticmethod
    def get_file_names(dir, suffix, limit=None):
        filenames = [filename for filename in sorted(os.listdir(dir)) if filename.endswith(suffix)]
        filenames = sorted(filenames)
        if limit is not None:
            filenames = filenames[0:limit]
        return filenames

    @staticmethod
    def get_key(img1_name: str, img2_name: str):
        return "{}_{}".format(img1_name, img2_name)

    def find_img_pair_from_imgs(self, img1_name, img2_name):
        return self.find_img_pair_from_key(SceneInfo.get_key(img1_name, img2_name))

    def find_img_pair_from_key(self, key):
        for diff in range(len(self.img_pairs_lists)):
            if self.img_pairs_maps[diff].__contains__(key):
                return self.img_pairs_maps[diff][key], diff
        return None

    @staticmethod
    def read_scene(scene_name, type="orig"):
        if type == "orig":
            Timer.start_check_point("reading scene info")
            print("scene={}".format(scene_name))
            img_pairs_lists, img_pairs_maps = read_image_pairs(scene_name)
            lazy = True
            img_info_map = read_images(scene_name, lazy=lazy)
            cameras = read_cameras(scene_name)
            Timer.end_check_point("reading scene info")
            return SceneInfo(img_pairs_lists, img_pairs_maps, img_info_map, cameras, scene_name, type="orig")
        elif type == "google":
            return read_google_scene(scene_name)
        else:
            raise Exception("unexpected type: {}".format(type))

    def get_camera_from_img(self, img: str):
        return self.cameras[self.img_info_map[img].camera_id]

    def imgs_for_comparing_difficulty(self, difficulty, suffix=".npy"):
        interesting_imgs = set()
        for img_pair in self.img_pairs_lists[difficulty]:
            interesting_imgs.add(img_pair.img1 + suffix)
            interesting_imgs.add(img_pair.img2 + suffix)
        return sorted(list(interesting_imgs))


def read_image_pairs(scene):

    file_name = "original_dataset/{}/{}_image_pairs.txt".format(scene, scene)
    f = open(file_name, "r")

    img_pairs_maps = [None] * 18
    img_pairs_lists = [None] * 18
    for i in range(18):
        img_pairs_maps[i] = {}
        img_pairs_lists[i] = []

    for line in f:
        bits = line.split(" ")
        img1 = bits[0].strip()[:-4]
        img2 = bits[1].strip()[:-4]
        diff = int(bits[2])
        img_pair = ImagePairEntry(img1, img2, diff)

        img_pairs_lists[img_pair.difficulty].append(img_pair)
        key = "{}_{}".format(img_pair.img1, img_pair.img2)
        img_pairs_maps[img_pair.difficulty][key] = img_pair

    return img_pairs_lists, img_pairs_maps


def read_images(scene, lazy=True):

    file_name = "original_dataset/{}/0/images.txt".format(scene)
    f = open(file_name, "r")

    image_map = {}
    odd = True
    for line in f:
        if line.strip().startswith("#"):
            continue
        bits = line.split(" ")
        if odd:
            odd = False
            image_id = int(bits[0])
            qw = float(bits[1])
            qx = float(bits[2])
            qy = float(bits[3])
            qz = float(bits[4])
            qs = (qw, qx, qy, qz)
            tx = float(bits[5])
            ty = float(bits[6])
            tz = float(bits[7])
            ts = (tx, ty, tz)
            camera_id = int(bits[8])
            name = bits[9].strip()[:-4]

            image_map[name] = ImageEntry(name, image_id, camera_id, qs, ts)
            # {
            #     "image_id": image_id,
            #     "camera_id": camera_id,
            #     "qs": (qw, qx, qy, qz),
            #     "t": (tx, ty, tz),
            # }

        else:
            odd = True
            # data = np.fromstring(line.strip(), dtype=float, sep=" ")
            # data = data.reshape((data.shape[0]//3, 3))
            # ["data"] = data

            if not lazy:
                image_map[name].read_data_from_line(line.strip())

    f.close()
    return image_map


def read_cameras(scene):

    file_name = "original_dataset/{}/0/cameras.txt".format(scene)
    f = open(file_name, "r")

    camera_map = {}

    for line in f:
        if line.strip().startswith("#"):
            continue
        bits = line.split(" ")
        id = int(bits[0])
        model = bits[1].strip()
        width = int(bits[2])
        height = int(bits[3])
        focal_length = float(bits[4])
        principal_point_x = int(bits[5])
        principal_point_y = int(bits[6])
        distortion = float(bits[7])
        camera_map[id] = CameraEntry(id, model, (height, width), focal_length, (principal_point_x, principal_point_y), distortion)

    f.close()
    return camera_map


def show_imgs_reuse(scene_info):

    img_pairs_per_diff = [len(diff_list) for diff_list in scene_info.img_pairs_lists]
    img_used = sum(img_pairs_per_diff) * 2
    img_all = len(scene_info.img_info_map)
    print("an img is used (at least) {} times on average".format(img_used / img_all))


def test():
    scene_info = SceneInfo.read_scene("scene1")
    show_imgs_reuse(scene_info)
    print("cameras and images read")


if __name__ == "__main__":
    test()