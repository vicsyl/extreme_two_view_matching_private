import torch.nn as nn
import torch
import kornia as K
from typing import Dict

"""
DISCLAIMER: Code adopted from Kornia's class LAFAffNetShapeEstimator (https://github.com/kornia/kornia/blob/master/kornia/feature/affine_shape.py)
"""

urls: Dict[str, str] = {"affnet": "https://github.com/ducha-aiki/affnet/raw/master/pretrained/AffNet.pth"}


class DenseAffNet(nn.Module):
    """Module, which extracts patches using input images and local affine frames (LAFs).
    Then runs AffNet on patches to estimate LAFs shape. This is based on the original code from paper
    "Repeatability Is Not Enough: Learning Discriminative Affine Regions via Discriminability"".
    See :cite:`AffNet2018` for more details.
    Then original LAF shape is replaced with estimated one. The original LAF orientation is not preserved,
    so it is recommended to first run LAFAffineShapeEstimator and then LAFOrienter.
    Args:
        pretrained: Download and set pretrained weights to the model.
    """

    def __init__(self, pretrained: bool=False, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 3, kernel_size=8, stride=1, padding=0, bias=True),
            nn.Tanh(),
            # nn.AdaptiveAvgPool2d(1),
        )
        self.patch_size = 32
        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(
                urls['affnet'], map_location=lambda storage, loc: storage
            )
            self.load_state_dict(pretrained_dict['state_dict'], strict=False)
        self.eval()
        print("DenseAffNet device: {}".format(device))
        if self.device == torch.device('cuda'):
            print("dense affnet cuda")
            self.cuda()
            self.features.cuda()
        else:
            self.cpu()

    @staticmethod
    def _normalize_input(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Utility function that normalizes the input by batch."""
        sp, mp = torch.std_mean(x, dim=(-3, -2, -1), keepdim=True)
        # WARNING: we need to .detach() input, otherwise the gradients produced by
        # the patches extractor with F.grid_sample are very noisy, making the detector
        # training totally unstable.
        return (x - mp.detach()) / (sp.detach() + eps)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: shape [Bx1xHxW]
        Returns:
            laf_out shape [BxNx2x3]
        """
        xy = self.features(self._normalize_input(img))
        BB, CH, HH, WW = xy.shape
        assert BB == 1

        xy = xy.permute(0, 2, 3, 1).view(-1, 3)
        # a1, a2 -> paper
        a1 = torch.cat([1.0 + xy[:, 0].reshape(-1, 1, 1), 0 * xy[:, 0].reshape(-1, 1, 1)], dim=2)
        a2 = torch.cat([xy[:, 1].reshape(-1, 1, 1), 1.0 + xy[:, 2].reshape(-1, 1, 1)], dim=2)
        new_laf_no_center = torch.cat([a1, a2], dim=1).reshape(-1, 1, 2, 2)
        N = new_laf_no_center.size(0)
        new_laf = torch.cat([new_laf_no_center, torch.zeros((N, 1, 2, 1), device=self.device)], dim=3)
        ellipse_scale = K.feature.get_laf_scale(new_laf)
        laf_out_flat = K.feature.scale_laf(K.feature.make_upright(new_laf), 1.0 / ellipse_scale)
        laf_out_flat = laf_out_flat.permute(1, 0, 2, 3)
        laf_out = laf_out_flat.reshape(HH, WW, 2, 3)
        return laf_out
