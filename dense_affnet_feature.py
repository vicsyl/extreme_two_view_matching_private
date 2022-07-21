from typing import Optional, Tuple
from affnet_clustering import affnet_clustering, affnet_clustering_torch
from affnet import affnet_rectify
from affnet import KptStruct
from hard_net_descriptor import HardNetDescriptor
from dense_affnet import DenseAffNet
import kornia.feature as KF

import torch

import cv2 as cv


def get_default_config():

    return {
        "affnet_no_clustering": False,
        "affnet_covering_type": "dense_cover",
        "affnet_covering_fraction_th": 0.95,
        "affnet_covering_max_iter": 100,

        "affnet_dense_affnet_use_orienter": False,
        "affnet_dense_affnet_enforce_connected_components": False,

        "affnet_include_all_from_identity": True
    }


class DenseAffnetFeature:

    def __init__(self, device: torch.device = torch.device('cpu'), config=get_default_config()):

        self.device = device
        self.config = config
        n_features = None
        sift_octave_layers = 3
        sift_contrast_threshold = 0.04
        sift_edge_threshold = 10
        sift_sigma = 1.6
        sift_detector = cv.SIFT_create(n_features, sift_octave_layers, sift_contrast_threshold, sift_edge_threshold, sift_sigma)
        self.hard_net = HardNetDescriptor(sift_detector, compute_laffs=True, filter=None, device=self.device)
        self.dense_affnet = DenseAffNet(pretrained=True, device=self.device)

    def forward(self,
                img: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
                                                              torch.Tensor,
                                                              torch.Tensor]:

        img_data = affnet_clustering_torch(img=img,
                                           dense_affnet=self.dense_affnet,
                                           mask=mask,
                                           use_cuda=self.device == torch.device("cuda"),
                                           conf=self.config)

        kpts_struct: KptStruct = affnet_rectify(img_name=None,
                                                hardnet_descriptor=self.hard_net,
                                                img_data=img_data,
                                                conf_map=self.config,
                                                mask=mask)

        scales = KF.get_laf_scale(kpts_struct.reprojected_laffs)
        scaled_laffs = KF.scale_laf(kpts_struct.reprojected_laffs, 1. / scales)

        ret = scaled_laffs, scales[:, :, 0], kpts_struct.descs[None]
        return ret
