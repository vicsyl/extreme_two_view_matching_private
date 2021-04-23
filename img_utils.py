import time

import matplotlib.pyplot as plt


def show_and_save_normal_clusters(normals, clustered_normals, normal_indices):

    cluster_color_names = ["red", "green", "blue"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Normals clustering" + str(time.time()))

    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.set_zlabel("z")

    ax.plot(0, 0, 0, 'o', color="black", markersize=2.0)

    rel_normals = normals[normal_indices == 3]
    ax.plot((rel_normals[:, 0]), (rel_normals[:, 1]), (rel_normals[:, 2]), '.', color="yellow", markersize=0.5)

    for i in range(len(clustered_normals)):
        rel_normals = normals[normal_indices == i]
        ax.plot((rel_normals[:, 0]), (rel_normals[:, 1]), (rel_normals[:, 2]), '.', color=cluster_color_names[i], markersize=0.5)

    for i in range(len(clustered_normals)):
        ax.plot((clustered_normals[i, 0]), (clustered_normals[i, 1]), (clustered_normals[i, 2]), 'o', color="black", markersize=5.0)

    ax.view_init(elev=0.1, azim=-40)

    x_lim = [-1, 1]
    y_lim = [-1, 1]
    z_lim = [-1, 1]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    plt.show(block=False)


def show_normals_components(normals, title, figsize=None):

    if len(normals.shape) == 5:
        normals = normals.squeeze(dim=0).squeeze(dim=0)

    img = normals.numpy()
    fig = plt.figure()
    plt.title(title)
    for index in range(3):
        # row, columns, index
        ax = fig.add_subplot(131 + index)
        ax.imshow(img[:, :, index])
    plt.show()
