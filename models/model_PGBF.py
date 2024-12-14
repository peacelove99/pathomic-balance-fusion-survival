import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.array_api import squeeze
from torch.onnx.symbolic_opset9 import square
from torch_geometric.nn import GlobalAttention

from models.model_CMTA import Transformer_P, Transformer_G
from models.model_MCAT import MultiheadAttention
from models.model_MOTCat import OT_Attn_assem
from models.model_tmi2024 import MultiHeadAttention
from models.model_utils import SNN_Block, Attn_Net_Gated


###########################
### PGBF Implementation ###
###########################
class PGBF_Surv(nn.Module):
    def __init__(self,
                 model_size: str = 'small', omic_sizes=None,
                 topk=30,
                 ot_impl="pot-uot-l2", ot_reg=0.05, ot_tau=0.5,
                 n_classes=4, coattn_model="CMTA", use_graph=False, dropout=0.25):
        super(PGBF_Surv, self).__init__()

        ### Genomic Embedding
        if omic_sizes is None:
            omic_sizes = []
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        hidden = self.size_dict_omic[model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=dropout))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        ### Pathology Embedding
        self.size_dict_path = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        hidden = self.size_dict_path[model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(dropout))
        self.pathology_fc = nn.Sequential(*fc)

        ### Encoder
        self.encoder_path = Transformer_P(feature_dim=hidden[-1])
        self.decoder_path = Transformer_P(feature_dim=hidden[-1])
        self.encoder_omic = Transformer_G(feature_dim=hidden[-1])
        self.decoder_omic = Transformer_G(feature_dim=hidden[-1])

        ### path graph WiKG
        self.use_graph = use_graph
        self.W_head = nn.Linear(256, 256)
        self.W_tail = nn.Linear(256, 256)
        self.scale = 256 ** -0.5
        self.topk = topk
        self.linear1 = nn.Linear(256, 256)
        self.linear2 = nn.Linear(256, 256)
        self.activation = nn.ReLU()
        self.message_dropout = nn.Dropout(dropout)
        att_net = nn.Sequential(nn.Linear(256, 256 // 2), nn.LeakyReLU(), nn.Linear(256 // 2, 1))
        self.readout = GlobalAttention(att_net)

        ### coattn graph & omic
        self.coattn_model = coattn_model
        if self.coattn_model == "TMI_2024":
            self.coattn = MultiHeadAttention(in_features=256, head_num=8)  # TMI_2024
        elif self.coattn_model == "MOTCat":
            self.coattn = OT_Attn_assem(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)  # MOTCat
        elif self.coattn_model == "MCAT":
            self.coattn = MultiheadAttention(embed_dim=256, num_heads=8)  # MCAT

        ### CMTA Attention
        self.P_in_G_Att = MultiheadAttention(embed_dim=256, num_heads=1)  # P->G Attention
        self.G_in_P_Att = MultiheadAttention(embed_dim=256, num_heads=1)  # G->P Attention

        ### path decoder
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=256, D=256, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])

        ### omic decoder
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=256, D=256, dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout)])

        ### Fusion Layer
        self.mm = nn.Sequential(*[nn.Linear(256 * 2, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()])

        ### Survival Layer
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, **kwargs):
        # LUAD Genomic Dimensions [89, 334, 534, 471, 1510, 482]
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, 7)]  # 包含六个不同组学的特征数据
        x_path = kwargs['x_path']

        ### Embedding
        # print('Embedding')
        # Genomic Embedding
        h_omic = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic)  # [6, 256]
        # print('h_omic_bag.size():', h_omic_bag.size())
        # Pathology Embedding
        h_path_bag = self.pathomics_fc(x_path)  # [num_patch, 256]
        # print('h_path_bag.size():', h_path_bag.size())

        ### Encoder
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(h_omic_bag)
        # Pathology Encoder
        cls_token_pathology_encoder, patch_token_pathology_encoder = self.pathomics_encoder(h_path_bag)

        ### path graph
        if self.use_graph:
            # print('path graph')
            h_path_bag = h_path_bag.unsqueeze(0)  # [1, num_patch, 256]
            # WiKG 公式1 每个补丁的嵌入投影为头部和尾部嵌入
            e_h = self.W_head(h_path_bag)  # embedding_head [num_patch, 256]
            e_t = self.W_tail(h_path_bag)  # embedding_tail [num_patch, 256]
            # print('e_h.size():', e_h.size(), ';e_t.size():', e_t.size())

            # WiKG 公式2 3 相似性得分最高的前 k 个补丁被选为补丁 i 的邻居
            attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 计算 e_h 和 e_t 之间的相似性(点积)
            # print('attn_logit.size():', attn_logit.size())
            topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # 获取 Top - k 注意力分数和对应索引
            # print('topk_weight.size():', topk_weight.size(), 'topk_index.size():', topk_index.size())
            topk_prob = F.softmax(topk_weight, dim=-1)  # 归一化注意力分数
            # print('topk_prob.size():', topk_prob.size())

            topk_index = topk_index.to(torch.long)  # 转换索引类型
            # print('topk_index.size():', topk_index.size())
            topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # 扩展索引以匹配 e_t 维度
            # print('topk_index_expanded.size():', topk_index_expanded.size())
            batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # 创建批次索引辅助张量
            # print('batch_indices.size():', batch_indices.size())
            Nb_h = e_t[batch_indices, topk_index_expanded, :]  # 使用索引从 e_t 中提取特征向量 neighbors_head
            # print('Nb_h.size():', Nb_h.size())

            # WiKG 公式 4 为有向边分配嵌入 embedding head_r
            eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))
            # print('eh_r.size():', eh_r.size())

            # WiKG 公式 6 计算 加权因子
            e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
            # print('e_h_expand.size():', e_h_expand.size())
            gate = torch.tanh(e_h_expand + eh_r)
            # print('gate.size():', gate.size())
            ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)
            # print('ka_weight.size():', ka_weight.size())

            # WiKG 公式 7 对 加权因子 进行归一化
            ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
            # print('ka_prob.size():', ka_prob.size())

            # WiKG 公式 5 计算补丁 i 相邻 N （i） 的尾部嵌入的线性组合
            e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)
            # print('e_Nh.size():', e_Nh.size())

            # WiKG 公式 8 将聚合的邻居信息 e_Nh 与原始 head 融合
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            # print('sum_embedding.size():', sum_embedding.size())
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            # print('bi_embedding.size():', bi_embedding.size())
            e_h = sum_embedding + bi_embedding
            # print('e_h.size():', e_h.size())

            # WiKG 公式 9 生成 graph-level 嵌入 embedding_graph
            e_h = self.message_dropout(e_h)
            # print('e_h.size():', e_h.size())
            e_g = self.readout(e_h.squeeze(0), batch=None)
            # print('e_g.size():', e_g.size())

        ### coattn graph & omic
        # print(self.coattn_model, 'coattn')
        if self.coattn_model == "TMI_2024":
            # h_omic_bag & h_path_bag = [batch_size, seq_len, in_feature]
            # print("q=h_omic_bag.unsqueeze(0):",h_omic_bag.unsqueeze(0).size())
            # print("k=e_h:",e_h.size())
            # print("v=e_h:",e_h.size())
            # h_path_bag, G_coattn = self.coattn(h_omic_bag.squeeze(0), h_path_bag.squeeze(0), h_path_bag.squeeze(0))  # TMI_2024
            h_path_bag, G_coattn = self.coattn(h_omic_bag.unsqueeze(0), e_h, e_h)  # TMI_2024
        elif self.coattn_model == "MOTCat":
            # A_coattn, _ = self.coattn(h_path_bag.unsqueeze(1), h_omic_bag.unsqueeze(1))  # MOTCat
            A_coattn, _ = self.coattn(e_h.permute(1, 0, 2), h_omic_bag.unsqueeze(1))  # MOTCat
            # print('A_coattn.size():', A_coattn.size())
            # h_path_coattn = torch.mm(A_coattn.squeeze(), h_path_bag.squeeze()).unsqueeze(1)
            h_path_coattn = torch.mm(A_coattn.squeeze(), e_h.squeeze()).unsqueeze(1)
            # print('h_path_coattn.size():', h_path_coattn.size())
        elif self.coattn_model == "MCAT":
            # h_path_coattn, A_coattn = self.coattn(h_omic_bag.unsqueeze(1), h_path_bag.unsqueeze(1), h_path_bag.unsqueeze(1))  # MCAT
            h_path_coattn, A_coattn = self.coattn(h_omic_bag.unsqueeze(1), e_h.permute(1, 0, 2), e_h.permute(1, 0, 2))  # MCAT
        elif self.coattn_model == "CMTA":
            pathology_in_genomics, Att = self.P_in_G_Att(
                patch_token_pathology_encoder.transpose(1, 0),
                patch_token_genomics_encoder.transpose(1, 0),
                patch_token_genomics_encoder.transpose(1, 0),)  # ([14642, 1, 256])
            genomics_in_pathology, Att = self.G_in_P_Att(
                patch_token_genomics_encoder.transpose(1, 0),
                patch_token_pathology_encoder.transpose(1, 0),
                patch_token_pathology_encoder.transpose(1, 0),)  # ([7, 1, 256])

        ### path decoder
        # print('path decoder')
        if self.coattn_model == "TMI_2024":
            A_path, h_path = self.path_attention_head(h_path_bag)  # TMI_2024
            A_path = A_path.squeeze(0)
            h_path = h_path.squeeze(0)
            # print('A_path.size():', A_path.size(), 'h_path.size():', h_path.size())
            A_path = torch.transpose(A_path, 1, 0)
            # print('A_path.size():', A_path.size())
            h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
            # print('h_path.size():', h_path.size())
            h_path = self.path_rho(h_path).squeeze()
            # print('h_path.size():', h_path.size())
        elif self.coattn_model == "MOTCat" or self.coattn_model == "MCAT":
            h_path_trans = self.path_transformer(h_path_coattn)  # MCAT & MOTCat
            # print('h_path_trans.size():', h_path_trans.size())
            A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
            # print('A_path.size():', A_path.size(), 'h_path.size():', h_path.size())
            A_path = torch.transpose(A_path, 1, 0)
            # print('A_path.size():', A_path.size())
            h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
            # print('h_path.size():', h_path.size())
            h_path = self.path_rho(h_path).squeeze()
            # print('h_path.size():', h_path.size())
        elif self.coattn_model == "CMTA":
            cls_token_pathology_decoder, _ = self.decoder_path(pathology_in_genomics.transpose(1, 0))

        ### omic decoder
        # print('omic decoder')
        if self.coattn_model == "MOTCat" or self.coattn_model == "MCAT":
            h_omic_trans = self.omic_transformer(h_omic_bag)  # MCAT & MOTCat
            A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
            A_omic = torch.transpose(A_omic, 1, 0)
            h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
            h_omic = self.omic_rho(h_omic).squeeze()
            # print(self.coattn_model, 'h_omic.size():', h_omic.size())
        elif self.coattn_model == "CMTA":
            cls_token_genomics_decoder, _ = self.decoder_omic(genomics_in_pathology.transpose(1, 0))

        ### Fusion Layer
        # print('Fusion Layer')
        if self.coattn_model == "TMI_2024":
            h = h_path
        elif self.coattn_model == "MOTCat" or self.coattn_model == "MCAT":
            h = self.mm(torch.cat([h_path, h_omic], axis=0))  # MCAT & MOTCat
        elif self.coattn_model == "CMTA":
            h = self.mm(
                torch.concat((
                        (cls_token_pathology_encoder + cls_token_pathology_decoder) / 2,
                        (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
                    ), dim=1,))
        # print(self.coattn_model, 'h.size():', h.size())

        ### Survival Layer
        # print('Survival Layer')
        # logits = self.classifier(h)  # .unsqueeze(0) # logits needs to be a [1 x 4] vector
        logits = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector
        # print('logits.size():', logits.size())
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        # print('Y_hat.size():', Y_hat.size())
        hazards = torch.sigmoid(logits)
        # print('hazards.size():', hazards.size())
        S = torch.cumprod(1 - hazards, dim=1)
        # print('S.size():', S.size())

        logits_omic = self.classifier(h_omic).unsqueeze(0)
        hazards_omic = torch.sigmoid(logits_omic)
        S_omic = torch.cumprod(1 - hazards_omic, dim=1)

        logits_path = self.classifier(h_path).unsqueeze(0)
        hazards_path = torch.sigmoid(logits_path)
        S_path = torch.cumprod(1 - hazards_path, dim=1)

        result = {'hazards': hazards_omic, 'S': S_omic}
        # result_omic = {'hazards': hazards_omic, 'S': S_omic}
        # result_path = {'hazards': hazards_path, 'S': S_path}
        result_omic = {'encoder': cls_token_pathology_encoder, 'decoder': cls_token_pathology_decoder}
        result_path = {'encoder': cls_token_genomics_encoder, 'decoder': cls_token_genomics_decoder}

        # attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}

        # return hazards, S, Y_hat, attention_scores, h_path, h_omic
        return result, result_omic, result_path

