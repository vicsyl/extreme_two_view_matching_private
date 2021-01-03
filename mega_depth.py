import numpy as np
import os

if __name__ == "__main__":

    os.open("scene1/cameras")

    directory = "/Users/vaclav/ownCloud/SVP/project/original_dataset/scene1/images"

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            print("{}/{}".format(directory, filename))
        #     f = open(filename)
        #     lines = f.read()
        #     print(lines[10])
        #     continue
        # else:
        #     continue

    depth = np.loadtxt('work/demo_mega_depth.txt', delimiter=',')

    print("done")
