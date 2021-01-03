import torch
import math

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