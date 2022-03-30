import cv2 as cv
import kornia as K
import kornia.feature as KF
from kornia.utils import batched_forward
from kornia_moons.feature import *
import numpy as np
import torch

import dense_hard_net
from utils import Timer
import numpy.ma as ma

"""
DISCLAIMER: taken from https://github.com/kornia/kornia-examples/blob/master/MKD_TFeat_descriptors_in_kornia.ipynb
"""

class HardNetDescriptor:

    new_version = True

    def __init__(self, feature_detector, device: torch.device=torch.device('cpu')):
        self.feature_detector = feature_detector
        if self.new_version:
            self.hardnet = dense_hard_net.DenseHardNet()
        else:
            #self.hardnet = KF.HardNet(True)
            self.hardnet = dense_hard_net.NormalHardNet()
        self.device = device
        self.affine = KF.LAFAffNetShapeEstimator(True)
        self.orienter = KF.LAFOrienter(32, angle_detector=KF.OriNet(True))
        self.set_device_eval_to_nets([self.hardnet, self.affine, self.orienter], self.device)

    @staticmethod
    def set_device_eval_to_nets(nets: list, device):
        for net in nets:
            net.eval()
            if device == torch.device('cuda'):
                net.cuda()
            else:
                net.cpu()

    def detectAndCompute(self, img, mask=None, give_laffs=False, filter=None, dense=False):
        Timer.start_check_point("HadrNet.detectAndCompute")
        # NOTE this is just how it was called before
        assert mask is None
        if dense:
            x_grid = torch.linspace(0, img.shape[1] - 1).to(self.device)
            x, y = torch.meshgrid(x, x)
        else:
            mask = np.ones(img.shape[:2], np.uint8)
            #mask = ma.masked_all((img.shape[1] - 1, img.shape[0] - 1))

            step_size = 1
            rx = ry = 100
            kps_mask_kp = [cv.KeyPoint(x, y, step_size) for y in range(0, ry, step_size)
                  for x in range(0, rx, step_size)]

            kps = self.feature_detector.detect(img, None)
            if filter is not None:
                kps = kps[::filter]
        ret = self.get_local_descriptors(img, kps, compute_laffs=give_laffs)
        if len(ret) != 2:
            # corner case
            descs = np.zeros(0)
            laffs = np.zeros(0)
        else:
            descs, laffs = ret

        Timer.end_check_point("HadrNet.detectAndCompute")
        if give_laffs:
            return kps, descs, laffs
        else:
            return kps, descs

    def get_local_descriptors(self, img, cv2_sift_kpts, compute_laffs=False):
        if len(cv2_sift_kpts) == 0:
            return np.array([])

        # We will not train anything, so let's save time and memory by no_grad()
        with torch.no_grad():
            #self.hardnet.eval()
            if len(img.shape) == 3:
                pass # OK
            elif len(img.shape) == 2:
                img = img.reshape(img.shape[0], img.shape[1], 1)
                img = np.repeat(img, 3, axis=2)
            else:
                raise Exception("Unexpected shape of the img: {}".format(img.shape))
            timg = K.color.rgb_to_grayscale(K.image_to_tensor(img, False).float() / 255.).to(self.device)
            lafs1 = laf_from_opencv_SIFT_kpts(cv2_sift_kpts, device=self.device)

            if compute_laffs:
                # We will estimate affine shape of the feature and re-orient the keypoints with the OriNet
                # self.affine.eval()
                # orienter.eval()
                lafs_aff = self.affine(lafs1, timg)
                lafs_to_use = self.orienter(lafs_aff, timg)

                for cv2_sift_kpt in cv2_sift_kpts:
                    cv2_sift_kpt.angle = 0.0
                    cv2_sift_kpt.class_id = -1
                    cv2_sift_kpt.octave = 0
                    cv2_sift_kpt.response = 0.0
                    cv2_sift_kpt.size = 2.0
                lafs2 = laf_from_opencv_SIFT_kpts(cv2_sift_kpts, device=self.device)

                lafs_aff2 = self.affine(lafs2, timg)
                lafs_to_use2 = self.orienter(lafs_aff2, timg)
                print()
            else:
                lafs_to_use = lafs1

            patches = KF.extract_patches_from_pyramid(timg, lafs_to_use, 32)

            B, N, CH, H, W = patches.size()

            if self.new_version:
                #patches = torch.permute(torch.from_numpy(img), dims=[2, 0, 1])[None]
                patches = timg
                # B = patches.size[0]
                # N = 1
                descs1 = batched_forward(self.hardnet, patches, self.device, 64)
                # descs1:T(1, 128, 473, 263) = 124399
                descs = torch.flatten(descs1[0], start_dim=1).permute((1, 0))


            else:

                patches = patches.view(B * N, CH, H, W)
                # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
                # So we need to reshape a bit :)
                # descs = self.hardnet(patches).view(B * N, -1)
                #descs1 = batched_forward(self.hardnet, patches, self.device, 128)
                descs1 = batched_forward(self.hardnet, patches, self.device, 64)
                descs = descs1.view(B * N, -1)

        return descs.detach().cpu().numpy(), lafs_to_use.detach().cpu()
