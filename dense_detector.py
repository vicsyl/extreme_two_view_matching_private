import torch


class DenseDetector:

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device

    def detect(self, img, filter=None):
        assert filter is None, "filter not implemented for {}".format(self.__class__.__name__)

        x_l_sp = torch.linspace(0, img.shape[1] - 1).to(self.device)
        y_l_sp = torch.linspace(0, img.shape[0] - 1).to(self.device)
        x, y = torch.meshgrid(x_l_sp, y_l_sp)
