import torch.nn.functional as F
import kornia.feature as KF
import torch


class NormalHardNet(KF.HardNet):

    def __init__(self):
        KF.HardNet.__init__(self, pretrained=True)

    # def forward(self, input: torch.Tensor) -> torch.Tensor:
    #     x_norm: torch.Tensor = self._normalize_input(input)
    #     x_features: torch.Tensor = self.features(x_norm)
    #     x_out = x_features.view(x_features.size(0), -1)
    #     ret = F.normalize(x_out, dim=1)
    #     return ret


class DenseHardNet(KF.HardNet):

    def __init__(self):
        KF.HardNet.__init__(self, pretrained=True)

    """Version of the HardNet to performs dense descriptor extraction
    """
    def forward(self, x):

        b, ch, h, w = x.shape
        x = x.float() / 255.
        x_ = x
        if ch > 1:
            x_ = x.mean(dim=1, keepdim=True)
        x_norm: torch.Tensor = self._normalize_input(x_)
        x_features: torch.Tensor = self.features(x_norm)
        ret = F.normalize(x_features, dim=1)
        return ret
