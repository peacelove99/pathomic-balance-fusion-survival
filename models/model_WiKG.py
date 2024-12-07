import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False))


class PGBF(nn.Module):
    def __init__(self,
                 model_size_omic: str = 'small', omic_sizes=[89, 334, 534, 471, 1510, 482],
                 dim_in=384, dim_hidden=512, topk=6, dropout=0.3):
        super(PGBF, self).__init__()

        ### Constructing Genomic SNN
        self.omic_sizes = omic_sizes
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}

        hidden = self.size_dict_omic[model_size_omic]  # 隐藏层大小
        sig_networks = []  # 存储所有处理基因的网络模块
        for input_dim in omic_sizes:  # omic_sizes=[100, 200, 300, 400, 500, 600]
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]  # 第一层
            for i, _ in enumerate(hidden[1:]):  # 遍历 hidden 中的后续维度，构建层与层之间的连接
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))  # 将网络 fc_omic 添加到 sig_networks
        self.sig_networks = nn.ModuleList(sig_networks)  # 存储多个神经网络模块


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

    def forward(self, **kwargs):

        # 处理基因数据 生成omic的嵌入
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]
        e_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        e_omic = torch.stack(e_omic).unsqueeze(1)  # [6, 1, 256]

        # 处理病理图像数据 生成graph的嵌入
        x_path = kwargs['x_path']
        print('x_path.size(0):', x_path.size())  # [num_patch, 1024]
        # WiKG 公式1 每个补丁的嵌入投影为头部和尾部嵌入
        x_path = self._fc1(x_path).unsqueeze(0)  # [1, num_patch, dim_hidden]
        x_path = (x_path + x_path.mean(dim=1, keepdim=True)) * 0.5  # 使特征分布更加平滑，有助于训练稳定性
        e_h = self.W_head(x_path)  # embedding_head [num_patch, dim_hidden]
        e_t = self.W_tail(x_path)  # embedding_tail [num_patch, dim_hidden]
        print('e_h.size():', e_h.size(),';e_t.size():', e_t.size())

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
        e_h = self.message_dropout(e_h) # [1, num_patch, dim_hidden]
        e_g = self.readout(e_h.squeeze(0), batch=None)  # [1, dim_hidden]

        return e_omic, e_h, e_g


if __name__ == "__main__":
    # data_WSI = torch.randn((1, 10000, 384)).to(device)
    data_WSI = torch.randn((6240, 1024)).to(device)
    data_omic1 = torch.randn(89).to(device)
    data_omic2 = torch.randn(334).to(device)
    data_omic3 = torch.randn(534).to(device)
    data_omic4 = torch.randn(471).to(device)
    data_omic5 = torch.randn(1510).to(device)
    data_omic6 = torch.randn(482).to(device)
    data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
    data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
    data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
    data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
    data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
    data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
    model = PGBF(dim_in=1024, dim_hidden=256).to(device)
    e_omic, e_h, e_g = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
    print('e_omic_bag.shape:', e_omic.shape)
    print('e_h.shape:', e_h.shape)
    print('e_g.shape:', e_g.shape)
