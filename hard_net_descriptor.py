import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import torch.nn.functional as F
from kornia.utils import batched_forward
from kornia_moons.feature import *

from affnet import show_sets_of_linear_maps
from transforms import get_rectification_rotations
from utils import Timer, timer_label_decorator
from transforms import decompose_homographies, homographies_jacobians

"""
DISCLAIMER: taken from https://github.com/kornia/kornia-examples/blob/master/MKD_TFeat_descriptors_in_kornia.ipynb
"""


class HardNetDescriptor:

    def __init__(self, sift_descriptor, compute_laffs, filter=None, device: torch.device=torch.device('cpu')):
        self.sift_descriptor = sift_descriptor
        self.hardnet = KF.HardNet(True)
        self.device = device
        self.affine = KF.LAFAffNetShapeEstimator(True)
        self.orienter = KF.LAFOrienter(32, angle_detector=KF.OriNet(True))
        self.set_device_eval_to_nets([self.hardnet, self.affine, self.orienter], self.device)
        self.custom_normals = None
        self.custom_K = None
        self.filter = filter
        self.compute_laffs = compute_laffs

    @staticmethod
    def set_device_eval_to_nets(nets: list, device):
        for net in nets:
            net.eval()
            if device == torch.device('cuda'):
                net.cuda()
            else:
                net.cpu()

    @staticmethod
    def resample_normals_to_img_size(data_in, to_size):
        """
        :param data_in: torch.Tensor(H, W, 3)
        :param to_size: torch.Tensor(H2, W2)
        :return:
        """
        #NOTE I may want to check the aspect ratio
        if data_in.shape[0] < to_size[0]:
            data_in = data_in.permute(2, 0, 1)[None]
            upsampling = torch.nn.Upsample(size=to_size, mode='nearest')
            data_in = upsampling(data_in)[0].permute(1, 2, 0)
        elif data_in.shape[0] > to_size[0]:
            assert False, "not tested" # TODO not tested
            data_in = F.interpolate(data_in, size=to_size, scale_factor=None, mode='nearest', align_corners=None)
        return data_in

    # TODO I don't like this hack
    def set_normals(self, custom_normals, custom_K):
        self.custom_normals = custom_normals
        self.custom_K = custom_K

    @timer_label_decorator("HardNet.detectAndCompute")
    def detectAndCompute(self, img, mask=None, give_laffs=False):
        # NOTE this is just how it was called before (see SuperPoint.detectAndCompute)
        # assert mask is None

        #mask_to_apply = None #if mask is None else mask.numpy().astype(np.uint8)
        kps = self.sift_descriptor.detect(img, mask)
        if self.filter is not None:
            kps = kps[::self.filter]

        ret = self.get_local_descriptors(img, kps, give_laffs=give_laffs)
        if len(ret) != 2:
            # corner case
            descs = np.zeros(0)
            laffs = np.zeros(0)
        else:
            descs, laffs = ret

        if give_laffs:
            return kps, descs, laffs
        else:
            return kps, descs

    def get_local_descriptors(self, img, cv2_sift_kpts, give_laffs=False):
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

            Timer.start_check_point("HardNet.lafs_computation")
            if self.custom_normals is not None:
                lafs_to_use = self.get_lafs_from_normals(cv2_sift_kpts, timg)
                self.custom_normals = None
                self.custom_K = None
            else:
                lafs = laf_from_opencv_SIFT_kpts(cv2_sift_kpts, device=self.device)
                if self.compute_laffs or give_laffs:
                    # We will estimate affine shape of the feature and re-orient the keypoints with the OriNet
                    # self.affine.eval()
                    # orienter.eval()
                    lafs2 = self.affine(lafs, timg)
                    lafs_to_use = self.orienter(lafs2, timg)
                else:
                    lafs_to_use = lafs
            Timer.end_check_point("HardNet.lafs_computation")

            patches = KF.extract_patches_from_pyramid(timg, lafs_to_use, 32)

            B, N, CH, H, W = patches.size()
            patches = patches.view(B * N, CH, H, W)

            # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
            # So we need to reshape a bit :)
            # descs = self.hardnet(patches).view(B * N, -1)
            descs = batched_forward(self.hardnet, patches, self.device, 128).view(B * N, -1)

        return descs.detach().cpu().numpy(), lafs_to_use.detach().cpu()

    def get_Hs_from_custom_normals(self, cv2_sift_kpts, timg):

        kps_long = torch.tensor([[kp.pt[0] + 0.5, kp.pt[1] + 0.5] for kp in cv2_sift_kpts], dtype=torch.long, device=self.device)
        in_img_mask = kps_long[:, 0] >= 0
        in_img_mask = torch.logical_and(in_img_mask, kps_long[:, 0] < timg.shape[3])
        in_img_mask = torch.logical_and(in_img_mask, kps_long[:, 1] >= 0)
        in_img_mask = torch.logical_and(in_img_mask, kps_long[:, 1] < timg.shape[2])
        kps_long = kps_long[in_img_mask]

        normals = HardNetDescriptor.resample_normals_to_img_size(self.custom_normals, timg.shape[2:]).to(self.device)
        normals = normals[kps_long[:, 1], kps_long[:, 0]]

        Rs = get_rectification_rotations(normals, self.device)
        K_torch = torch.from_numpy(self.custom_K).to(dtype=torch.float32, device=self.device)
        Hs = K_torch @ Rs @ torch.inverse(K_torch)
        return Hs

    def batch_and_invert_affines(self, affines, cv2_sift_kpts, timg):

        affines = affines[None, :, :2, :]
        visualise = False
        if visualise:
            self.visualize_lafs(affines.clone(), cv2_sift_kpts, timg)
        affines[:, :, :, :2] = torch.inverse(affines[:, :, :, :2])
        return affines

    def update_translations_from_cv2(self, affines, cv2_sift_kpts):

        locations = torch.tensor([list(cv_kpt.pt) for cv_kpt in cv2_sift_kpts], device=self.device)
        affines[0, :, :, 2] = locations
        return affines

    def get_lafs_from_normals_old(self, cv2_sift_kpts, timg):

        Hs = self.get_Hs_from_custom_normals(cv2_sift_kpts, timg)
        # the problem is also that the translation component is not zero (but will be updated below)
        _, affines = decompose_homographies(Hs, self.device)
        affines = self.batch_and_invert_affines(affines, cv2_sift_kpts, timg)
        affines = self.update_translations_from_cv2(affines, cv2_sift_kpts)
        return affines

    def get_lafs_from_normals(self, cv2_sift_kpts, timg):

        Hs = self.get_Hs_from_custom_normals(cv2_sift_kpts, timg)
        points_hom = torch.tensor([(*cv2_sift_kpt.pt, 1) for cv2_sift_kpt in cv2_sift_kpts], device=self.device)
        affines = homographies_jacobians(Hs, points_hom, self.device)
        affines = self.batch_and_invert_affines(affines, cv2_sift_kpts, timg)
        affines[:, :, :, 2] = points_hom[:, :2]
        return affines

    def visualize_lafs(self, affines2x3, cv2_sift_kpts, timg):

        # TEST
        lafs_test = laf_from_opencv_SIFT_kpts(cv2_sift_kpts, device=self.device)
        lafs2_test = self.affine(lafs_test, timg)
        lafs_to_use_test = self.orienter(lafs2_test, timg)

        lafs_to_use_test = lafs_to_use_test[:, :, :, :2]
        lafs_to_use_inv_vis = torch.inverse(lafs_to_use_test)

        # SHOW ME
        affines2x3 = affines2x3[:, :, :, :2]
        show_sets_of_linear_maps([affines2x3, lafs_to_use_inv_vis], label="both")
        show_sets_of_linear_maps([affines2x3], label="affine maps from depth maps")
        show_sets_of_linear_maps([lafs_to_use_inv_vis], label="lafs from AffNet")
