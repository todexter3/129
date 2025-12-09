import os
import torch
from models import PatchTST_C_c, PatchTST_C_group, PatchTST_C_group_speed, PatchTST_C_group_mult_speed, MLP


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            
            'PatchTST_C_group': PatchTST_C_group,
            'PatchTST_C_group_speed': PatchTST_C_group_speed,
            'PatchTST_C_group_mult_speed': PatchTST_C_group_mult_speed,
            'PatchTST_C_c':PatchTST_C_c,
            'MLP':MLP

        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        pass
        # if args.use_gpu:
        #     # os.environ["CUDA_VISIBLE_DEVICES"] = str(
        #     #     args.gpu) if not args.use_multi_gpu else args.devices
        #     device = torch.device('cuda')
        #     print('Use GPU: cuda:{}'.format(args.gpu))
        # else:
        #     device = torch.device('cpu')
        #     print('Use CPU')
        # return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
