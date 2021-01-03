import numpy as np

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
            tx = float(bits[5])
            ty = float(bits[6])
            tz = float(bits[7])
            camera_id = int(bits[8])
            name = bits[9].strip()

            image_map[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qs": (qw, qx, qy, qz),
                "t": (tx, ty, tz),
            }

        else:
            odd = True
            data = np.fromstring(line.strip(), dtype=float, sep=" ")
            data = data.reshape((data.shape[0]//3, 3))
            image_map[name]["data"] = data

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
        camera_map[id] = {
            "model": model,
            "width": width,
            "height": height,
            "focal_length": focal_length,
            "principal_point_x": principal_point_x,
            "principal_point_y": principal_point_y,
            "distortion": distortion,
        }

    f.close()
    return camera_map

if __name__ == "__main__":

    cameras = read_cameras("scene1")
    images = read_images("scene1")
    print("cameras and images read")