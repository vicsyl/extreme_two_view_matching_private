import numpy as np
from dataclasses import dataclass
from utils import Timer

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

    def get_input_dir(self):
        return "original_dataset/{}/images".format(self.name)

    def get_img_file_path(self, img_name):
        return '{}/{}.jpg'.format(self.get_input_dir(), img_name)

    def get_img_K(self, img_name):
        camera_id = self.img_info_map[img_name].camera_id
        return self.cameras[camera_id].get_K()

    def find_img_pair(self, key):
        for diff in range(len(self.img_pairs_lists)):
            if self.img_pairs_maps[diff].__contains__(key):
                return self.img_pairs_maps[diff][key], diff
            # for img_pair_entry in self.img_pairs_lists[diff]:
            #     key_img_pe = "{}_{}".format(img_pair_entry.img1, img_pair_entry.img2)
            #     if key_img_pe == key:
            #         return img_pair_entry, diff
        return None

    @staticmethod
    def read_scene(scene_name, lazy=True):
        Timer.start_check_point("reading scene info")
        print("scene={}, lazy={}".format(scene_name, lazy))
        img_pairs_lists, img_pairs_maps = read_image_pairs(scene_name)
        img_info_map = read_images(scene_name, lazy=lazy)
        cameras = read_cameras(scene_name)
        Timer.end_check_point("reading scene info")
        return SceneInfo(img_pairs_lists, img_pairs_maps, img_info_map, cameras, scene_name)

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


def read_images(scene, lazy=False):

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


def test():
    cameras = read_cameras("scene1")
    images = read_images("scene1")
    img_pairs_lists, img_pairs_maps = read_image_pairs("scene1")
    print("cameras and images read")
