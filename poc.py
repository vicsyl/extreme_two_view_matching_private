import torch
import math
import cv2 as cv
from matplotlib import pyplot as plt

def sobel():
    img = cv.imread('original_dataset/scene1/images/frame_0000000010_3.jpg', None)

    laplacian = cv.Laplacian(img, cv.CV_64F)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    kernels = cv.getDerivKernels(1, 0, 5)

    plt.show()

if __name__ == "__main__":


    c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)

    d = torch.cat(3*[torch.norm(c, dim=0)])
    d_kept = torch.cat(3*[torch.norm(c, dim=0, keepdim=True)])

    width_linspace = torch.linspace(-540, 539, steps=1080)
    height_linspace = torch.linspace(-960, 1919, steps=1920)

    grid_y, grid_x = torch.meshgrid(height_linspace, width_linspace)

    distances_from_origin = (grid_x / 1000) ** 2 + (grid_y / 1000) ** 2

    distances_from_origin_real = torch.sqrt(distances_from_origin)
    print()