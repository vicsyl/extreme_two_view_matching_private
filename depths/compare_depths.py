import cv2 as cv
import glob
import torch
import matplotlib.pyplot as plt

from utils import read_depth_data_from_path


def imshow(data, title):
    plt.figure()
    plt.imshow(data)
    plt.title(title)
    plt.show()
    plt.close()


def compare_depths_from_path(path, img_file_name):

    img_name = path.split("/")[-1][:-4]

    img = cv.imread(img_file_name)
    plt.imshow(img)
    plt.title(img_name)
    plt.show()
    plt.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_path = path[:-4] + "_orig.npy"
    orig_depth_data = read_depth_data_from_path(orig_path, height=None, width=None, device=device)
    orig_depth_data = orig_depth_data[0, 0]

    depth_data = read_depth_data_from_path(path, height=None, width=None, device=device)
    depth_data = depth_data[0, 0]

    scale_orig = (orig_depth_data / depth_data).flatten()
    valid_orig_pxs = (orig_depth_data != 0.0).sum().item()
    scale_median = torch.sort(scale_orig)[0][int(scale_orig.shape[0] - valid_orig_pxs / 2)].item()

    depth_data = torch.where(orig_depth_data == 0.0, 0.0, depth_data)

    B = orig_depth_data.flatten().unsqueeze(1)
    A = depth_data.flatten().unsqueeze(1)
    scale_lstsq = torch.lstsq(B, A).solution[0].item()
    print("scale median " + str(scale_median))
    print("scale lst sq " + str(scale_lstsq))
    # depth_data = depth_data * scale_median
    depth_data = depth_data * scale_lstsq

    data_diff = depth_data - orig_depth_data

    #imshow((orig_depth_data != 0.0), "valid mask for " + img_name)
    imshow(depth_data, "scaled depth data " + img_name)
    imshow(orig_depth_data, "original depth data for " + img_name)
    #imshow(data_diff, "diff for " + img_name)
    data_diff = data_diff.unsqueeze(0).expand(3, -1, -1)

    abs_max = torch.abs(data_diff).max()
    data_diff = data_diff / abs_max
    max1 = data_diff.min()
    max2 = data_diff.max()

    data_diff2 = data_diff.clone()
    data_diff2 = data_diff2.permute(1, 2, 0)
    coords = torch.where(orig_depth_data != 0.0)
    flattened_diff = data_diff2[coords[0], coords[1]]
    variance = torch.var(flattened_diff)
    print(variance)

    #
    data_diff[0] = torch.where(data_diff[0] < 0, -data_diff[0], 0.0)
    data_diff[1] = torch.where(data_diff[1] < 0, -data_diff[1], 0.0)
    data_diff[1] = torch.where(data_diff[1] > 0, data_diff[1], 0.0)
    data_diff[2] = torch.where(data_diff[2] > 0, data_diff[2], 0.0)
    data_diff = data_diff.permute(1, 2, 0)
    #imshow(data_diff, "diff for " + img_name)
    imshow(data_diff, "diff for {},\n values from {} to {},\n variance {}".format(img_name, max1, max2, variance))

    #
    # print("depth data read from " + path + " and " + orig_path)


def main():

    input_base = "/Users/vaclav/ownCloud/diploma_thesis/project/depth_data/megadepth_ds/depths"
    img_base = "/Users/vaclav/ownCloud/diploma_thesis/project/depth_data/megadepth_ds/imgs"

    glob_str = input_base + "/*_o.npy"
    paths_glob = glob.glob(glob_str)

    if paths_glob is None:
        raise Exception("wrong path:" + glob_str)
    if len(paths_glob) == 0:
        raise Exception("no data in path: " + glob_str)

    for path in paths_glob:
        img_file_name = "{}/{}.jpg".format(img_base, path.split("/")[-1][:-4])
        compare_depths_from_path(path, img_file_name)


if __name__ == "__main__":
    main()
    print("That's it")
