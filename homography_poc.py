from matplotlib import pyplot as plt
import numpy as np
from scene_info import *

def meshgrid(w, h, w_res, h_res):

    x = np.linspace(0, w - 1, w_res).astype(np.int)
    y = np.linspace(0, h - 1, h_res).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def main():

    cameras = read_cameras()
    first_key = next(iter(cameras.keys()))
    camera = cameras[first_key]

    meshgrid(50, 50)

    img = np.zeros((1920, 1080, 3), dtype=np.uint8)
    #img[:, :, 0] = 255
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()