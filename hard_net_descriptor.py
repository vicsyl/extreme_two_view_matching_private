import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
import numpy as np
import torch

"""
DISCLAIMER: taken from https://github.com/kornia/kornia-examples/blob/master/MKD_TFeat_descriptors_in_kornia.ipynb
"""


class HardNetDescriptor:

    def __init__(self, sift_descriptor, device: torch.device=torch.device('cpu')):
        self.sift_descriptor = sift_descriptor
        self.hardnet = KF.HardNet(True)
        self.device = device
        if self.device == torch.device('cuda'):
            self.hardnet.cuda()
        else:
            self.hardnet.cpu()

    def detectAndCompute(self, img, mask):
        # NOTE this is just how it was called before
        assert mask is None
        kps = self.sift_descriptor.detect(img, None)
        descs = self.get_local_descriptors(img, kps)
        return kps, descs

    def get_local_descriptors(self, img, cv2_sift_kpts):
        if len(cv2_sift_kpts) == 0:
            return np.array([])

        # We will not train anything, so let's save time and memory by no_grad()
        with torch.no_grad():
            self.hardnet.eval()
            timg = K.color.rgb_to_grayscale(K.image_to_tensor(img, False).float() / 255.)
            lafs = laf_from_opencv_SIFT_kpts(cv2_sift_kpts, device=self.device)
            patches = KF.extract_patches_from_pyramid(timg, lafs, 32)
            B, N, CH, H, W = patches.size()
            # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
            # So we need to reshape a bit :)
            descs = self.hardnet(patches.view(B * N, CH, H, W)).view(B * N, -1)
        return descs.detach().cpu().numpy()
