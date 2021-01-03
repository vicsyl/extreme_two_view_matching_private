import torch
import math

if __name__ == "__main__":

    width_linspace = torch.linspace(-540, 539, steps=1080)
    height_linspace = torch.linspace(-960, 1919, steps=1920)

    grid_y, grid_x = torch.meshgrid(height_linspace, width_linspace)

    distances_from_origin = (grid_x / 1000) ** 2 + (grid_y / 1000) ** 2

    distances_from_origin_real = torch.sqrt(distances_from_origin)
    print()