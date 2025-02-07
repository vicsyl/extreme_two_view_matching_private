import matplotlib.pyplot as plt
import numpy as np
import glob
import h5py
import os

from dataclasses import dataclass
from utils import Timer, quaternions_to_R
from img_utils import show_or_close


def add_google_image(img_name, image_info_map, calibration_file):

    if not image_info_map.__contains__(img_name):
        h5_data = h5py.File(calibration_file, "r")
        K = h5_data["K"][()]
        R = h5_data["R"][()]
        T = h5_data["T"][()]
        q = h5_data["q"][()]
        image_info_map[img_name] = ImageEntry(img_name, image_id=None, camera_id=None, qs=q, t=T, R=R, K=K)


def read_google_scene(scene_name, file_name_suffix, show_first=0):

    img_pairs_lists = {}
    img_pairs_maps = {}
    image_info_map = {}

    for diff in range(10):

        img_pairs_lists[diff] = []
        img_pairs_maps[diff] = {}

        print("Diff: {}".format(diff))
        file_name = "{}/{}/set_100/new-vis-pairs/keys-th-0.{}.npy".format(SceneInfo.base_dir, scene_name, diff)
        data_np = np.load(file_name)

        # just to check first couple of tuples in the "easiest" level
        if diff == 0:
            cache = data_np

        counter = 0
        for i in range(data_np.shape[0]):

            img_name_tuple = data_np[i].split("-")
            img1_path = "{}/{}/set_100/images/{}{}".format(SceneInfo.base_dir, scene_name, img_name_tuple[0], file_name_suffix)
            img1_exists = os.path.isfile(img1_path)
            img2_path = "{}/{}/set_100/images/{}{}".format(SceneInfo.base_dir, scene_name, img_name_tuple[1], file_name_suffix)
            img2_exists = os.path.isfile(img2_path)
            cal1_path = "{}/{}/set_100/calibration/calibration_{}.h5".format(SceneInfo.base_dir, scene_name, img_name_tuple[0])
            cal1_exists = os.path.isfile(cal1_path)
            cal2_path = "{}/{}/set_100/calibration/calibration_{}.h5".format(SceneInfo.base_dir, scene_name, img_name_tuple[1])
            cal2_exists = os.path.isfile(cal2_path)
            if img1_exists and img2_exists and cal1_exists and cal2_exists:
                counter = counter + 1

                entry = ImagePairEntry(img_name_tuple[0], img_name_tuple[1], diff)
                img_pairs_maps[diff][data_np[i]] = entry
                img_pairs_lists[diff].append(entry)

                add_google_image(img_name_tuple[0], image_info_map, cal1_path)
                add_google_image(img_name_tuple[1], image_info_map, cal2_path)

                if i < show_first and (diff == 9):
                    plt.figure(figsize=(10, 10))
                    plt.subplot(1, 2, 1)
                    img = plt.imread("{}/{}/set_100/images/{}{}".format(SceneInfo.base_dir, scene_name, img_name_tuple[0], file_name_suffix))
                    plt.title("{} from difficulty {}".format(i, diff))
                    plt.imshow(img)
                    plt.subplot(1, 2, 2)
                    img = plt.imread("{}/{}/set_100/images/{}{}".format(SceneInfo.base_dir, scene_name, img_name_tuple[1], file_name_suffix))
                    plt.title("{} from difficulty {}".format(i, diff))
                    plt.imshow(img)
                    show_or_close(True)
            else:
                print("WARNING: something missing for {}: {}, {}, {}, {}".format(img_name_tuple, img1_path, img2_path, cal1_path, cal2_path))


        print("{} valid pairs for diff {}".format(counter, diff))

    return SceneInfo(img_pairs_lists, img_pairs_maps, image_info_map, cameras=None, name=scene_name, type="google", file_name_suffix=file_name_suffix)


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
    file_name_suffix: str

    base_dir = "."

    def get_input_dir(self):
        if self.type == "orig":
            return "{}/original_dataset/{}/images".format(SceneInfo.base_dir, self.name)
        elif self.type == "google":
            return "{}/{}/set_100/images".format(SceneInfo.base_dir, self.name)
        else:
            raise Exception("unexpected type: {}".format(self.type))

    def get_img_file_path(self, img_name):
        if self.file_name_suffix is None:
            print("WARNING: file_name_suffix not set")
            self.file_name_suffix = ".jpg"
        return '{}/{}{}'.format(self.get_input_dir(), img_name, self.file_name_suffix)

    def get_img_K(self, img_name, img):
        img_entry = self.img_info_map[img_name]
        if img_entry.K is not None:
            K_to_scale = img_entry.K
        else:
            K_to_scale = self.cameras[img_entry.camera_id].get_K()

        K_to_scale[:2, :] *= img.shape[1] / (K_to_scale[0, 2] * 2.0)
        assert abs(K_to_scale[0, 2] * 2 - img.shape[1]) < 1.0
        assert abs(K_to_scale[1, 2] * 2 - img.shape[0]) < 1.0
        return K_to_scale

    def depth_input_dir(self):
        if self.type == "orig":
            return "{}/depth_data/mega_depth/{}".format(SceneInfo.base_dir, self.name)
        elif self.type == "google":
            return "{}/depth_data/{}".format(SceneInfo.base_dir, self.name)
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

    @staticmethod
    def get_key_from_pair(img_pair: ImagePairEntry):
        return SceneInfo.get_key(img_pair.img1, img_pair.img2)

    def find_img_pair_from_imgs(self, img1_name, img2_name):
        return self.find_img_pair_from_key(SceneInfo.get_key(img1_name, img2_name))

    def find_img_pair_from_key(self, key):
        for diff in range(len(self.img_pairs_lists)):
            if self.img_pairs_maps[diff].__contains__(key):
                return self.img_pairs_maps[diff][key], diff
        return None

    @staticmethod
    def read_scene(scene_name, type="orig", file_name_suffix=".jpg", show_first=0):
        if type == "orig":
            Timer.start_check_point("reading scene info")
            print("scene={}".format(scene_name))
            img_pairs_lists, img_pairs_maps = read_image_pairs(scene_name)
            lazy = True
            img_info_map = read_images(scene_name, lazy=lazy)
            cameras = read_cameras(scene_name)
            Timer.end_check_point("reading scene info")
            file_name_suffix = ".png" if scene_name[-1] in ["6", "7", "8"] else ".jpg"
            return SceneInfo(img_pairs_lists, img_pairs_maps, img_info_map, cameras, scene_name, type="orig", file_name_suffix=file_name_suffix)
        elif type == "google":
            return read_google_scene(scene_name, file_name_suffix, show_first)
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

    file_name = "{}/original_dataset/{}/{}_image_pairs.txt".format(SceneInfo.base_dir, scene, scene)
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

    file_name = "{}/original_dataset/{}/0/images.txt".format(SceneInfo.base_dir, scene)
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

            image_map[name] = ImageEntry(name, image_id, camera_id, qs, ts, R=None, K=None)
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

    file_name = "{}/original_dataset/{}/0/cameras.txt".format(SceneInfo.base_dir, scene)
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


def calculate_overall_accuracy():

    scene = SceneInfo.read_scene("edinburgh", type="google", show_first=0)

# estimate_k = False
#  rectify = True
    s = """0   0.04664938911514254
1   0.4222972972972973
2   0.40202702702702703
3   0.39864864864864863
4   0.40202702702702703
5   0.36486486486486486
6   0.39864864864864863
7   0.40202702702702703
8   0.375
9   0.40878378378378377"""
    # overall: 0.27843223459511335

# estimate_k = False
#   rectify = False
#     s = """0   0.056576576576576575
# 1   0.4850498338870432
# 2   0.47840531561461797
# 3   0.49169435215946844
# 4   0.4850498338870432
# 5   0.4717607973421927
# 6   0.48172757475083056
# 7   0.4750830564784053
# 8   0.49169435215946844
# 9   0.48172757475083056"""
    # overall: 0.33821554985963626

    accs = [float(line.split("   ")[1].strip()) for line in s.split("\n")]

    all = 0
    acc = 0.0
    for i in range(10):
        assert 0 < accs[i] < 1
        l = len(scene.img_pairs_lists[i])
        all = all + l
        acc = acc + accs[i] * l
    acc = acc / all
    print("Overall acurracy: {}".format(acc))


def show():

    scene = SceneInfo.read_scene("edinburgh", type="google", show_first=0)

    matching_pairs = [
        "5fd2c91236049babb753c6be7d99cdc_cb706ca5ce5c4625b7b29f1cced2418f",
    "e153533ee59843e0bce5072cd9c13a35_423d856c619a42a3aa341e41972bcc18",
    "f34c59d050448395101ac297d2fcf6_29c203e30ddd461cab3ed7a0225b0171",
    "d023d86287e04da0a7239b93e7dd260c_b65d3668d39e46ca9330a2abeb7c5285",
    "cb706ca5ce5c4625b7b29f1cced2418f_4a8e1cbfb45a45fc9e872b0e6b564741",
    "a4a010b9ff7143b0aaee9f53e000b3e6_7ce20ade05044dffb0220dd2dcb31628",
    "a3c549a16b4981bb400614da79ccac_0a2cccf4559a422bbb3f59c555793f31",
    "d36717005c624a678ac1cfa9eb5a5190_c92027d4a43e45d0822e4fd932b9d1f1",
    "d1f0979446ea68291618a2dc175_0a2cccf4559a422bbb3f59c555793f31",
    "c92027d4a43e45d0822e4fd932b9d1f1_57050d1f0979446ea68291618a2dc175",
    ]

    # matching_pairs = [
    #     "b0695cf27e3a45699cabafa926b6ad3f-2c18960517b24892a5ffb52341ece9eb",
    #     "8d377076df88437daccbf07cdf4f3e5a-0b1ea4bbfbeb45428631265fb85ab89b",
    #     "48767b964d9b49078e4015d02c98ce76-01bbad62912c4bde8586a16786c35db5",
    #     "f5fd2c91236049babb753c6be7d99cdc-01522fc97a924bf1adec026152c6305f",
    #     "cdd3fd3f2b4948438b715e1f7963cf77-9c453410e2ed4bd0bd288d7bf2308fe3",
    #     "2c18960517b24892a5ffb52341ece9eb-2b5315968bc5468c995b978620879439",
    #     "fe9eff017cbc45dca2ad0a7037b16e6b-d36717005c624a678ac1cfa9eb5a5190",
    #     "ec28d890df844db48a3d20f5bd8ea80b-01522fc97a924bf1adec026152c6305f",
    #     "9ccf06e9e3fe42f4933c09a55e632395-043f6f1c4dba43a5b99a81ff3b8780e8",
    #     "e4c503ff3448479db49f90d833a8e2b6-043f6f1c4dba43a5b99a81ff3b8780e8",
    # ]

    for pair in matching_pairs:
        pair = pair.replace("_", "-")
        found = False
        for i in range(9, 10):
            m = scene.img_pairs_maps[i]
            if m.__contains__(pair):
                found = True
                entry = m[pair]
                plt.figure(figsize=(10, 10))
                plt.title("pair: {}".format(pair))
                plt.subplot(1, 2, 1)
                img = plt.imread("{}/set_100/images/{}{}".format(scene.name, entry.img1, scene.file_name_suffix))
                plt.imshow(img)
                plt.subplot(1, 2, 2)
                img = plt.imread("{}/set_100/images/{}{}".format(scene.name, entry.img2, scene.file_name_suffix))
                #plt.title("{} from difficulty {}".format(i, diff))
                plt.imshow(img)
                plt.show()
                break
        if not found:
            print("{} not found!!!".format(pair))


if __name__ == "__main__":
    scene = SceneInfo.read_scene("phototourism/st_peters_square", type="google", show_first=5)
    #test()
    #show()