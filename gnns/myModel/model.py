from turtle import forward
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv


class Attention(nn.Module):

    def __init__(self, hidden_dim, attn_drop):
        """语义层次的注意力

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        """
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain)
        nn.init.xavier_normal_(self.attn, gain)

    def forward(self, h):
        """
        :param h: tensor(N, M, d) 顶点基于不同元路径/类型的嵌入，N为顶点数，M为元路径/类型数
        :return: tensor(N, d) 顶点的最终嵌入
        """
        attn = self.attn_drop(self.attn)
        # (N, M, d) -> (M, d) -> (M, 1)
        w = torch.tanh(self.fc(h)).mean(dim=0).matmul(attn.t())
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((h.shape[0],) + beta.shape)  # (N, M, 1)
        z = (beta * h).sum(dim=1)  # (N, d)
        return z


class BiGraphEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        '''
        二分图编码器

        参数
        ----
        in_dim: 输入特征维数
        hidden_dim: 隐藏层特征维数
        attn_drop: 注意力dropout
        '''
        super().__init__()
        self.gcn = GraphConv(in_dim, hidden_dim, activation=nn.PReLU())
    
    def forward(self, g, feats):
        '''
        g: 输入的二分图, 只包含两种类型的顶点和一种类型的边
        feats: 图顶点特征 (N, in_dim)
        ----
        return: 经过一次gcn后的顶点特征
        '''
        h = self.gcn(g, feats) # (N, h_dim)
        return h


class BiGraphContrastLayer(nn.Module):

    '''
    二分图对比学习层

    '''
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = BiGraphEncoder(in_dim, hidden_dim)
        self.fuse = None
        self.loss = nn.BCELoss()

    def forward(self, g, feats):
        h_pos = self.encoder(g, feats)

        # 扰动
        g_neg = self.fuse(g)
        h_neg = self.encoder(g_neg, feats)


class Contrast(nn.Module):
    '''
    全图对比学习层
    in_dim: 输入维数
    hidden_dim: 隐藏层维数
    etypes: 图的关系三元组(src, edge_type, dst)
    '''
    def __init__(self, in_dim, hidden_dim, etypes):
        super().__init__()
        self.bigraphs = nn.ModuleDict({
            etype: BiGraphContrastLayer(in_dim, hidden_dim) for _, etype, _ in etypes
        })
