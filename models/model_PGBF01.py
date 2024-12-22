import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.array_api import squeeze
from torch.onnx.symbolic_opset9 import square
from torch_geometric.nn import GlobalAttention

from models.model_CMTA import Transformer_P, Transformer_G, MultiheadAttention_CMTA
from models.model_MCAT import MultiHeadAttention_MCAT
from models.model_MOTCat import OT_Attn_assem
from models.model_tmi2024 import MultiHeadAttention_TMI_2024
from models.model_utils import SNN_Block, Attn_Net_Gated

###########################
### PGBF Implementation ###
###########################
class PGBF_Surv01(nn.Module):
    def __init__(self, omic_sizes=None, ot_impl="pot-uot-l2",
                 n_classes=4, args=None
                 ):
        super(PGBF_Surv01, self).__init__()
        self.coattn_model = args.coattn_model  # ["MOTCat", "CMTA"]
        self.path_decoder = args.path_decoder  # [0, 1, 2, 3]
        self.omic_decoder = args.omic_decoder  # [0, 1, 2]
        self.fusion_layer = args.fusion_layer  # [0, 1]
        topk = args.topk  # [6, 12, 18, 24, 30]
        ot_reg = args.ot_reg  # [0.05, 0.1]
        ot_tau = 0.5  #
        dropout = args.dropout  # [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        ### Genomic Embedding
        if omic_sizes is None:
            omic_sizes = []
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=dropout))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        ### Pathology Embedding
        hidden = [1024, 256, 256]
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
        if self.coattn_model == "MOTCat":
            self.coattn = OT_Attn_assem(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)  # MOTCat
        elif self.coattn_model == "CMTA":
            self.P_in_G_Att = MultiheadAttention_CMTA(embed_dim=256, num_heads=6)  # P->G Attention
            self.G_in_P_Att = MultiheadAttention_CMTA(embed_dim=256, num_heads=6)  # G->P Attention

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
        # Pathology Embedding
        h_path_bag = self.pathology_fc(x_path)  # [num_patch, 256]

        ### Encoder
        # Genomic Encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.encoder_omic(h_omic_bag.unsqueeze(0))  # [1, 256]  [1, 6, 256]
        # Pathology Encoder
        # WiKG 公式1 每个补丁的嵌入投影为头部和尾部嵌入
        e_h = self.W_head(h_path_bag.unsqueeze(0))  # embedding_head [1, num_patch, 256]
        e_t = self.W_tail(h_path_bag.unsqueeze(0))  # embedding_tail [1, num_patch, 256]
        # WiKG 公式2 3 相似性得分最高的前 k 个补丁被选为补丁 i 的邻居
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 计算 e_h 和 e_t 之间的相似性(点积) [1, num_patch, num_patch]
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # 获取 Top - k 注意力分数和对应索引 [1, num_patch, topk]
        topk_prob = F.softmax(topk_weight, dim=-1)  # 归一化注意力分数 [1, num_patch, topk]
        topk_index = topk_index.to(torch.long)  # 转换索引类型 [1, num_patch, topk]
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # 扩展索引以匹配 e_t 维度 [1, num_patch, topk]
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # 创建批次索引辅助张量 [1, 1, 1]
        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # 使用索引从 e_t 中提取特征向量 neighbors_head [1, num_patch, topk, 256]
        # WiKG 公式 4 为有向边分配嵌入 embedding head_r
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))  # [1, num_patch, topk, 256]
        # WiKG 公式 6 计算 加权因子
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)  # [1, num_patch, topk, 256]
        gate = torch.tanh(e_h_expand + eh_r)  # [1, num_patch, topk, 256]
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)  # [1, num_patch, topk]
        # WiKG 公式 7 对 加权因子 进行归一化
        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)  # [1, num_patch, 1, topk]
        # WiKG 公式 5 计算补丁 i 相邻 N （i） 的尾部嵌入的线性组合
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)  # [1, num_patch, 256]
        # WiKG 公式 8 将聚合的邻居信息 e_Nh 与原始 head 融合
        sum_embedding = self.activation(self.linear1(e_h + e_Nh))  # [1, num_patch, 256]
        bi_embedding = self.activation(self.linear2(e_h * e_Nh))  # [1, num_patch, 256]
        e_h = sum_embedding + bi_embedding  # [1, num_patch, 256]
        # WiKG 公式 9 生成 graph-level 嵌入 embedding_graph
        patch_token_pathology_encoder = self.message_dropout(e_h)  # [1, num_patch, 256]
        cls_token_pathology_encoder = self.readout(patch_token_pathology_encoder.squeeze(0), batch=None)  # [1, 256]

        ### coattn graph & omic
        # print(self.coattn_model, 'coattn')
        omic_coattn = patch_token_genomics_encoder.transpose(1, 0)
        path_coattn = patch_token_pathology_encoder.transpose(1, 0)
        # print("omic_coattn.size():",omic_coattn.size())
        # print("path_coattn.size():",path_coattn.size())
        if self.coattn_model == "MOTCat":
            Att, _ = self.coattn(path_coattn, omic_coattn)  # [1, 1, 6, num_patch]
            # print('Attn.size():', Att.size())
            genomics_in_pathology = torch.mm(Att.squeeze(), patch_token_pathology_encoder.squeeze())  # [1, 6, num_patch]
            print('genomics_in_pathology.size():', genomics_in_pathology.size())
        elif self.coattn_model == "CMTA":
            pathology_in_genomics, Att = self.P_in_G_Att(path_coattn, omic_coattn, omic_coattn)  # [num_patch, 1, 256]
            # print('pathology_in_genomics.size():', pathology_in_genomics.size())
            # print('Attn.size():', Att.size())
            genomics_in_pathology, Att = self.G_in_P_Att(omic_coattn, path_coattn, path_coattn)  # [6, 1, 256]
            print('genomics_in_pathology.size():', genomics_in_pathology.size())
            print('Attn.size():', Att.size())

        ### path decoder
        # print('path decoder')
        # print("genomics_in_pathology.size():", genomics_in_pathology.size())
        path_decoder = self.path_decoder
        if path_decoder == 0:
            A_path, h_path = self.path_attention_head(genomics_in_pathology)  # TMI_2024
            # print('A_path.size():', A_path.size(), 'h_path.size():', h_path.size())
            A_path = A_path.squeeze(0)
            h_path = torch.mm(F.softmax(A_path.transpose(1, 0), dim=1), h_path.squeeze(0))  # [1, 256]
            cls_token_pathology_decoder = self.path_rho(h_path)  # [1,256]
        elif path_decoder == 1:
            h_path_trans = self.path_transformer(genomics_in_pathology)  # MCAT & MOTCat
            # print('h_path_trans.size():', h_path_trans.size())
            A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
            # print('A_path.size():', A_path.size(), 'h_path.size():', h_path.size())
            h_path = torch.mm(F.softmax(A_path.transpose(1, 0), dim=1), h_path)  # [1, 256]
            cls_token_pathology_decoder = self.path_rho(h_path)  # [1,256]
        elif path_decoder == 2:
            cls_token_pathology_decoder, _ = self.decoder_path(genomics_in_pathology.transpose(1, 0))  # [1, 256]
            # print('cls_token_pathology_decoder.size():', cls_token_pathology_decoder.size())
        elif path_decoder == 3:
            cls_token_pathology_decoder, _ = self.decoder_omic(genomics_in_pathology.transpose(1, 0))  # [1, 256]
            # print('cls_token_pathology_decoder.size():', cls_token_pathology_decoder.size())

        ### omic decoder
        if self.coattn_model == "CMTA":
            h_omic_bag = pathology_in_genomics
        # print('omic decoder')
        omic_decoder = self.omic_decoder
        if omic_decoder == 0:
            h_omic_trans = self.omic_transformer(h_omic_bag)  # MCAT & MOTCat [6, 256]
            A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))  # [6, 1] [6, 256]
            A_omic = torch.transpose(A_omic, 1, 0)  # [1, 6]
            h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)  # [1, 256]
            cls_token_genomics_decoder = self.omic_rho(h_omic)  # [1, 256]
        elif omic_decoder == 1:
            cls_token_genomics_decoder, _ = self.decoder_omic(h_omic_bag.transpose(1, 0))  # [1, 256]
        elif omic_decoder == 2:
            cls_token_genomics_decoder, _ = self.decoder_path(h_omic_bag.transpose(1, 0))  # [1, 256]

        ### Fusion Layer
        # print('Fusion Layer')
        fusion_layer = self.fusion_layer
        if fusion_layer == 0:
            h = cls_token_pathology_decoder
        elif fusion_layer == 1:
            h = self.mm(torch.cat([cls_token_pathology_decoder, cls_token_genomics_decoder], axis=-1))  # MCAT & MOTCat
        elif omic_encoder == 1 and path_encoder == 2:
            h = self.mm(
                torch.concat((
                        (cls_token_pathology_encoder + cls_token_pathology_decoder) / 2,
                        (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
                    ), dim=1,))
        # print(self.coattn_model, 'h.size():', h.size())

        ### Survival Layer
        # print('Survival Layer')
        logits = self.classifier(h) # logits needs to be a [1 x 4] vector
        # print('logits.size():', logits.size())
        Y_hat = torch.topk(logits, 1, dim=-1)[1]
        # print('Y_hat.size():', Y_hat.size())
        hazards = torch.sigmoid(logits)
        # print('hazards.size():', hazards.size())
        S = torch.cumprod(1 - hazards, dim=1)
        # print('S.size():', S.size())

        logits_omic = self.classifier(cls_token_genomics_decoder).unsqueeze(0)
        hazards_omic = torch.sigmoid(logits_omic)
        S_omic = torch.cumprod(1 - hazards_omic, dim=1)

        logits_path = self.classifier(cls_token_pathology_decoder).unsqueeze(0)
        hazards_path = torch.sigmoid(logits_path)
        S_path = torch.cumprod(1 - hazards_path, dim=1)

        result = {'hazards': hazards, 'S': S}
        result_omic = {'encoder': cls_token_genomics_encoder, 'decoder': cls_token_genomics_decoder}
        result_path = {'encoder': cls_token_pathology_encoder, 'decoder': cls_token_pathology_decoder}

        # attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}

        # return hazards, S, Y_hat, attention_scores, h_path, h_omic
        # return result, result_omic, result_path

