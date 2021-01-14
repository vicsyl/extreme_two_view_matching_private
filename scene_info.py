import numpy as np
from dataclasses import dataclass


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
    principal_point_x_y: (int, int)
    distortion: float


def read_image_pairs(scene):

    file_name = "original_dataset/{}/{}_image_pairs.txt".format(scene, scene)
    f = open(file_name, "r")

    ret = [None] * 18
    for i in range(18):
        ret[i] = []

    for line in f:
        bits = line.split(" ")
        img1 = bits[0].strip()[:-4]
        img2 = bits[1].strip()[:-4]
        diff = int(bits[2])
        img_pair = ImagePairEntry(img1, img2, diff)
        ret[img_pair.difficulty].append(img_pair)

    return ret


def read_images(scene):

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


if __name__ == "__main__":

    cameras = read_cameras("scene1")
    images = read_images("scene1")
    img_pairs = read_image_pairs("scene1")
    print("cameras and images read")