import matplotlib.pyplot as plt


def show_normals_components(normals, title, figsize=None):

    if len(normals.shape) == 5:
        normals = normals.squeeze(dim=0).squeeze(dim=0)

    img = normals.numpy()
    # img = normals.numpy() * 255
    # img[:, :, 2] = -img[:, :, 2] / 255
    fig = plt.figure()
    plt.title(title)
    for index in range(3):
        # row, columns, index
        ax = fig.add_subplot(131 + index)
        ax.imshow(img[:, :, index])
    plt.show()
