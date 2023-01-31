import torch
from models.NNET import NNET
import surface_normal_uncertainty.utils.utils as utils


class StandaloneConfig:

    def __init__(self, input_height: int, input_width: int, device: torch.device):
        self.architecture = "BN"
        self.pretrained = "scannet"
        self.sampling_ratio = 0.4
        self.importance_ratio = 0.7
        self.input_height = input_height
        self.input_width = input_width
        self.device = device


class SurfaceNormals:

    def __init__(self):
        self.models_map = {}

    def compute_normals(self, img, height, width, device):

        key = (height, width)
        if self.models_map.__contains__(key):
            model = self.models_map[key]
        else:
            print(f"creating new model for (h, w) = ({height}, {width})")
            model = self.get_model(StandaloneConfig(height, width, device))
            self.models_map[key] = model
            print(f"model created, number of models: {len(self.models_map)}")

        pred_norm, pred_uncertainty = self.compute_normals_inner(model, img)
        return pred_norm, pred_uncertainty

    def get_model(self, config: StandaloneConfig):

        # load checkpoint
        checkpoint = './checkpoints/%s.pt' % config.pretrained
        print('loading checkpoint... {}'.format(checkpoint))
        model = NNET(config).to(config.device)
        model = utils.load_checkpoint(checkpoint, model)
        model.eval()
        print('loading checkpoint... / done')

        return model

    def compute_normals_inner(self, model, img):

        with torch.no_grad():

            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]

            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            # to numpy arrays
            # img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1)[0]        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()
            pred_alpha = utils.kappa_to_alpha(pred_kappa)

        pred_uncertainty = pred_alpha
        return pred_norm, pred_uncertainty
