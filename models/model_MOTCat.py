import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

import ot

from models.model_utils import *
from torch_geometric.nn import GlobalAttention


class OT_Attn_assem(nn.Module):
    def __init__(self, impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)

    def normalize_feature(self, x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Parmas:
            weight1 : (N, D)
            weight2 : (M, D)

        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2) ** 2  # (N, M)

            src_weight = weight1.sum(dim=1) / weight1.sum()
            dst_weight = weight2.sum(dim=1) / weight2.sum()

            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(),
                               M=cost_map_detach / cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map * flow
            dist = torch.sum(dist)
            return flow, dist

        elif self.impl == "pot-uot-l2":
            a, b = torch.from_numpy(ot.unif(weight1.size()[0]).astype('float64')).to(weight1.device), torch.from_numpy(
                ot.unif(weight2.size()[0]).astype('float64')).to(weight2.device)
            self.cost_map = torch.cdist(weight1, weight2) ** 2  # (N, M)

            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach / cost_map_detach.max()

            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b,
                                                           M=M_cost.double(), reg=self.ot_reg, reg_m=self.ot_tau)
            flow = flow.type(torch.FloatTensor).cuda()

            dist = self.cost_map * flow  # (N, M)
            dist = torch.sum(dist)  # (1,) float
            return flow, dist

        else:
            raise NotImplementedError

    def forward(self, x, y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''
        x = x.squeeze()
        y = y.squeeze()

        x = self.normalize_feature(x)
        y = self.normalize_feature(y)

        pi, dist = self.OT(x, y)
        return pi.T.unsqueeze(0).unsqueeze(0), dist


#############################
### MOTCAT Implementation ###
#############################
class MOTCAT_Surv(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str = 'small', model_size_omic: str = 'small', dropout=0.25, ot_reg=0.1, ot_tau=0.5,
                 ot_impl="pot-uot-l2",
                 dim_in=1024, dim_hidden=256, topk=6):
        super(MOTCAT_Surv, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}

        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        ### construct graph WiKG
        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)
        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.linear1 = nn.Linear(dim_hidden, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
        att_net = nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(), nn.Linear(dim_hidden // 2, 1))
        self.readout = GlobalAttention(att_net)

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### OT-based Co-attention
        self.coattn = OT_Attn_assem(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout,
                                                        activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256 * 2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None

        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]

        h_path_bag = self.wsi_net(x_path).unsqueeze(1)  ### path embeddings are fed through a FC layer

        # 处理病理图像数据 生成graph的嵌入
        # WiKG 公式1 每个补丁的嵌入投影为头部和尾部嵌入
        x_path = self._fc1(x_path).unsqueeze(0)  # [1, num_patch, dim_hidden]
        x_path = (x_path + x_path.mean(dim=1, keepdim=True)) * 0.5  # 使特征分布更加平滑，有助于训练稳定性
        e_h = self.W_head(x_path)  # embedding_head [num_patch, dim_hidden]
        e_t = self.W_tail(x_path)  # embedding_tail [num_patch, dim_hidden]
        # print('e_h.size():', e_h.size(), ';e_t.size():', e_t.size())

        # WiKG 公式2 3 相似性得分最高的前 k 个补丁被选为补丁 i 的邻居
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 计算 e_h 和 e_t 之间的相似性(点积)
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # 获取 Top - k 注意力分数和对应索引
        topk_prob = F.softmax(topk_weight, dim=2)  # 归一化注意力分数

        topk_index = topk_index.to(torch.long)  # 转换索引类型
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # 扩展索引以匹配 e_t 维度
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # 创建批次索引辅助张量
        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # 使用索引从 e_t 中提取特征向量 neighbors_head

        # WiKG 公式 4 为有向边分配嵌入 embedding head_r
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))
        # WiKG 公式 6 计算 加权因子
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)
        # WiKG 公式 7 对 加权因子 进行归一化
        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        # WiKG 公式 5 计算补丁 i 相邻 N （i） 的尾部嵌入的线性组合
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)
        # WiKG 公式 8 将聚合的邻居信息 e_Nh 与原始 head 融合
        sum_embedding = self.activation(self.linear1(e_h + e_Nh))
        bi_embedding = self.activation(self.linear2(e_h * e_Nh))
        e_h = sum_embedding + bi_embedding
        # WiKG 公式 9 生成 graph-level 嵌入 embedding_graph
        e_h = self.message_dropout(e_h)  # [1, 23641, 256]
        # e_g = self.readout(e_h.squeeze(0), batch=None)  # [1, 512]
        e_g = self.readout(e_h.squeeze(0), batch=None).unsqueeze(0)  # [1, 1, 512]

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]  ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(1)  ### omic embeddings are stacked (to be used in co-attention)
        # print('h_path_bag.size():',h_path_bag.size()) # [21209, 1, 256]
        # print('e_h.permute(1, 0, 2).size():',e_h.permute(1, 0, 2).size()) # [21209, 1, 256]
        # print('e_g.size():',e_g.size()) # [1, 1, 256]
        ### Coattn
        # A_coattn, _ = self.coattn(h_path_bag, h_omic_bag)  # MOTCat
        A_coattn, _ = self.coattn(e_h.permute(1, 0, 2), h_omic_bag)  # MOTCat & WiKG
        # A_coattn, _ = self.coattn(e_g, h_omic_bag)  # MOTCat & WiKG readout
        # h_path_coattn = torch.mm(A_coattn.squeeze(), h_path_bag.squeeze()).unsqueeze(1)
        h_path_coattn = torch.mm(A_coattn.squeeze(), e_h.squeeze()).unsqueeze(1)

        ### Path
        h_path_trans = self.path_transformer(h_path_coattn)
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho(h_path).squeeze()

        ### Omic
        h_omic_trans = self.omic_transformer(h_omic_bag)
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()

        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))

        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}

        # return hazards, S, Y_hat, h_path, h_omic, attention_scores
        return hazards, S, Y_hat, attention_scores