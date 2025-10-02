import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm

import random
import torch.nn.functional as F
import dgl
from torch_geometric.nn import GCNConv,RGCNConv
import copy
from copy import deepcopy

from model_llama import Transformer_lar, ModelArgs

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

def rbf_kernel(x1, x2, gamma=1.0): #径向基函数核
    dist = torch.cdist(x1, x2)**2
    return torch.exp(-gamma * dist)   

def laplacian_kernel(x1, x2, gamma=None):  #拉普拉斯核
    if gamma is None:
        gamma = 1.0 / x1.size(1)  # 默认1/n_features
    dist = torch.cdist(x1, x2, p=1)
    return torch.exp(-gamma * dist)

def cosine_similarity_kernel(x1, x2):  #余弦相似度核
    x1_norm = torch.nn.functional.normalize(x1, p=2, dim=1)
    x2_norm = torch.nn.functional.normalize(x2, p=2, dim=1)
    return torch.mm(x1_norm, x2_norm.t())

def sigmoid_kernel(x1, x2, gamma=None, coef0=1): #Sigmoide核
    if gamma is None:
        gamma = 1.0 / x1.size(1)  # 默认1/n_features
    return torch.tanh(gamma * torch.mm(x1, x2.t()) + coef0)


# 对比学习损失函数
def contrastive_loss(z1, z2, temperature=0.05):
    """
    计算SimCSE的对比损失
    z1, z2: 正样本对的嵌入，形状为(batch_size, hidden_size)
    temperature: 温度参数，控制软化程度
    """
    kernel_matrix_1 = laplacian_kernel(z1, z2, gamma=0.1)  #核学习
    kernel_matrix_2 = cosine_similarity_kernel(z1, z2)  #核学习
    # target_similarity = labels.unsqueeze(1) == labels.unsqueeze(0)
    # target_similarity = target_similarity.float()

    batch_size = z1.shape[0]
    # 计算两个正样本间的相似度
    cos_sim = nn.functional.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2) / temperature
    # 对角线元素是正样本对的相似度
    labels = torch.arange(batch_size).long().to(z1.device)

    # loss = 0.2*nn.CrossEntropyLoss()(kernel_matrix_1, labels) + 0.8*nn.CrossEntropyLoss()(kernel_matrix_2, labels)
    loss = nn.CrossEntropyLoss()(kernel_matrix_2, labels)
    return loss


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(256, 256, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits, sc_1, sc_2
    
        
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0) 
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)


class M3LNet(nn.Module):
    def __init__(self, **config):
        super(M3LNet, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding, hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        self.disc = Discriminator(32 * 2)

        self.llama = Transformer_lar()

    def forward(self, bg_d, v_p, mode="train"):
        v_d_o, net_feats_pos_1, net_feats_pos_2, edge_feats_1, edge_feats_2, node_feats_1, node_feats_2 = self.drug_extractor(bg_d)  #(64, 290, 128)
        v_p_o = self.protein_extractor(v_p)  #(64, 1185, 128)
        #原来的样本
        f_o_sub, att_o = self.bcn(v_d_o, v_p_o)  #(64, 256)

        f_o_llm = f_o_sub.unsqueeze(1)
        f_o_llm = self.llama(f_o_llm).squeeze(0)#.permute(1, 0, 2)  #LLM optimizing

        # print("==========f_o_llm.shape===========",f_o_llm.shape)
        # print("==========f_o.shape===========",f_o.shape)
        beta = 0.1 
        f_o = beta *f_o_llm + (1-beta) * f_o_sub

        # f_o = f_o_sub

        score_o = self.mlp_classifier(f_o)

        #正负样本-net_view
        f_pos_1, att_pos_1 = self.bcn(net_feats_pos_1, v_p_o)
        f_pos_2, att_pos_2 = self.bcn(net_feats_pos_2, v_p_o)
        score_pos_1 = self.mlp_classifier(f_pos_1)
        score_pos_2 = self.mlp_classifier(f_pos_2)
        loss_contra_net = contrastive_loss(score_pos_1, score_pos_2)

        #正负样本-edge_view
        f_edge_1, att_edge_1 = self.bcn(edge_feats_1, v_p_o)
        f_edge_2, att_edge_2 = self.bcn(edge_feats_2, v_p_o)
        score_edge_1 = self.mlp_classifier(f_edge_1)
        score_edge_2 = self.mlp_classifier(f_edge_2)
        loss_contra_edge = contrastive_loss(score_edge_1, score_edge_2)

        #正负样本-node_view
        f_node_1, att_node_1 = self.bcn(node_feats_1, v_p_o)
        f_node_2, att_node_2 = self.bcn(node_feats_2, v_p_o)
        score_node_1 = self.mlp_classifier(f_node_1)
        score_node_2 = self.mlp_classifier(f_node_2)
        loss_contra_node = contrastive_loss(score_node_1, score_node_2)


        if mode == "train":
            # v_d_o, v_d_node_pos, v_d_node_neg, v_d_edge_neg = self.drug_extractor(bg_d)
            # v_p_o = self.protein_extractor(v_p)
            # #原来的样本
            # f_o, att_o = self.bcn(v_d_o, v_p_o)
            # score_o = self.mlp_classifier(f_o)

            # #正负样本
            # f_node_pos, att_node_pos = self.bcn(v_d_node_pos, v_p_o) #正样本
            # score_node_pos = self.mlp_classifier(f_node_pos)

            # f_node_neg, att_node_neg = self.bcn(v_d_node_neg, v_p_o) #负样本--node view
            # score_node_neg = self.mlp_classifier(f_node_neg)

            # f_edge_neg, att_edge_pos = self.bcn(v_d_edge_neg, v_p_o) #负样本--edge view
            # score_edge_neg = self.mlp_classifier(f_edge_neg)

            # logits_node_view, logits_node_view_pos, logits_node_view_neg = self.disc(f_node_pos, f_o, f_node_neg) #node view
            # logits_edge_view, logits_edge_view_pos, logits_edge_view_neg = self.disc(f_node_pos, f_o, f_edge_neg) #node view
                
            # #创建正负样本对
            # # logits_node_view = torch.cat((score_node_pos, score_node_neg), 1)  #node view
            # # logits_edge_view = torch.cat((f_node_pos, f_edge_neg), 1)  #edge view

            return v_d_o, v_p_o, f_o, score_o, loss_contra_net, loss_contra_edge, loss_contra_node
        
        elif mode == "eval":
            # v_d_o, _, _, _ = self.drug_extractor(bg_d)
            # v_p_o = self.protein_extractor(v_p)
            # f_o, att_o = self.bcn(v_d_o, v_p_o)
            # score_o = self.mlp_classifier(f_o)  

            return v_d_o, v_p_o, score_o, att_o



class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        self.read = AvgReadout()
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn1 = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.gnn2 = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation, dropout=[0.1,0.1,0.1])  #[0.3,0.3,0.3] [0.1,0.1,0.1]
        self.gnn3 = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation, dropout=[0.2,0.2,0.2])  #[0.5,0.5,0.5] [0.2,0.2,0.2]
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        #原来样本
        batch_graph_o = deepcopy(batch_graph)
        node_feats = batch_graph_o.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn1(batch_graph_o, node_feats)
        batch_size = batch_graph_o.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)

        #创建正样本对
        batch_graph_pos = deepcopy(batch_graph)
        node_feats_pos = batch_graph_pos.ndata.pop('h')
        node_feats_pos = self.init_transform(node_feats_pos)
        node_feats_pos_1 = self.gnn2(batch_graph_pos, node_feats_pos)
        node_feats_pos_2 = self.gnn3(batch_graph_pos, node_feats_pos)
        batch_size_pos = batch_graph_pos.batch_size
        net_feats_pos_1 = node_feats_pos_1.view(batch_size_pos, -1, self.output_feats)   
        net_feats_pos_2 = node_feats_pos_2.view(batch_size_pos, -1, self.output_feats)   

        #创建正负样本--node view
        batch_graph_neg_node = deepcopy(batch_graph) #获取node
        node_feats_neg = batch_graph_neg_node.ndata.pop('h')
        #random.shuffle(node_feats_neg)
        node_feats_neg = self.init_transform(node_feats_neg)
        node_feats_neg_1 = F.dropout(node_feats_neg, 0.1, training=True)
        node_feats_neg_2 = F.dropout(node_feats_neg, 0.2, training=True)
        node_feats_neg_1 = self.gnn1(batch_graph_neg_node, node_feats_neg_1)
        node_feats_neg_2 = self.gnn1(batch_graph_neg_node, node_feats_neg_2)
        batch_size_neg = batch_graph_neg_node.batch_size
        node_feats_1 = node_feats_neg_1.view(batch_size_neg, -1, self.output_feats) 
        node_feats_2 = node_feats_neg_2.view(batch_size_neg, -1, self.output_feats) 

        #创建正负样本--edge view
        batch_graph_neg_edge = deepcopy(batch_graph) #获取node
        batch_graph_neg_edge_1 = deepcopy(batch_graph) #获取node
        batch_graph_neg_edge_2 = deepcopy(batch_graph) #获取node
        edge_feats_neg = batch_graph_neg_edge.edata.pop('e')
        #random.shuffle(edge_feats_neg)
        edge_feats_neg_1 = F.dropout(edge_feats_neg, 0.1, training=True)
        edge_feats_neg_2 = F.dropout(edge_feats_neg, 0.2, training=True)
        batch_graph_neg_edge_1.edata['e'] = edge_feats_neg_1
        batch_graph_neg_edge_2.edata['e'] = edge_feats_neg_2
        node_edge_feats_neg_1 = batch_graph_neg_edge_1.ndata.pop('h')
        node_edge_feats_neg_2 = batch_graph_neg_edge_2.ndata.pop('h')
        node_edge_feats_neg_1 = self.init_transform(node_edge_feats_neg_1)
        node_edge_feats_neg_2 = self.init_transform(node_edge_feats_neg_2)
        node_edge_feats_neg_1 = self.gnn1(batch_graph_neg_edge_1, node_edge_feats_neg_1)
        node_edge_feats_neg_2 = self.gnn1(batch_graph_neg_edge_2, node_edge_feats_neg_2)
        batch_size_neg_edge = batch_graph_neg_edge.batch_size
        edge_feats_1 = node_edge_feats_neg_1.view(batch_size_neg_edge, -1, self.output_feats) 
        edge_feats_2 = node_edge_feats_neg_2.view(batch_size_neg_edge, -1, self.output_feats) 

        
        return node_feats, net_feats_pos_1, net_feats_pos_2, edge_feats_1, edge_feats_2, node_feats_1, node_feats_2  #node_feats_pos, node_feats_neg, node_edge_feats_neg


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda(device=device)
        self.random_matrix = [val.cuda(device=device) for val in self.random_matrix]
