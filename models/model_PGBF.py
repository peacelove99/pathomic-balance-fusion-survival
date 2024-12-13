import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *
from models.model_utils import SNN_Block


###########################
### PGBF Implementation ###
###########################
class PGBF_Surv(nn.Module):
    def __init__(self,
                 model_size_path: str = 'small',
                 model_size_omic: str = 'small', omic_sizes=[]):
        super(PGBF_Surv, self).__init__()

        ### omic encoder SNN
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

    def forward(self, **kwargs):
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]
        print(x_omic.__sizeof__())
        print(x_omic[0].size())

        ### omic encoder
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        print(h_omic.__sizeof__())
        print(h_omic[0].size())
        h_omic_bag = torch.stack(h_omic)
        print(h_omic_bag.__sizeof__())

        h_omic_bag = h_omic_bag.unsqueeze(1)
        # h_omic_bag = h_omic_bag.transpose(0, 1)  # tmi 2024

