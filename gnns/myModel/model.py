from turtle import forward
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv
import dgl
from gnns.utils.graph_fuse import fuse_graph
import torch.nn.functional as F


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
        # (N, M, d) -> (M, d) -> (M, 1) edge-scale attention 对每一种元路径/类型需要求一个整体的节点表示之后计算attention
        w = torch.tanh(self.fc(h)).mean(dim=0).matmul(attn.t())
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((h.shape[0],) + beta.shape)  # (N, M, 1)
        z = (beta * h).sum(dim=1)  # (N, d)
        return z


class BiGraphContrastLayer(nn.Module):

    '''
    二分图对比学习层
    ----
    参数
    :in_dim 输入特征维数
    :hidden_dim 图编码器特征输出维数 即隐藏层维数
    :predict_type 对比学习的顶点类型 边的目标顶点
    :fuse_way 图扰动方式
    :tem 温度系数
    '''
    def __init__(self, in_dim, hidden_dim, predict_type, fuse_way='drop-edge', tem=0.7):
        super().__init__()
        self.predict_type = predict_type
        # TODO GCN可以替换成GAT
        self.graph_encoder = GraphConv(in_dim, hidden_dim, norm='right', activation=nn.PReLU())
        self.fuse_way = fuse_way
        self.tem = tem

    def _calculate_loss(self, pos_g, pos_feat, neg_g, neg_feat, predict_type_id):
        pos_nodes = pos_g.filter_nodes(lambda nodes: (nodes.data['_TYPE'] == predict_type_id))
        # (N, hid_d)
        h_pos = pos_feat[pos_nodes]
        neg_nodes = neg_g.filter_nodes(lambda nodes: (nodes.data['_TYPE'] == predict_type_id))
        # (N, hid_d)
        h_neg = neg_feat[neg_nodes]

        # (N, 1)
        pos_loss = F.cosine_similarity(h_pos, h_neg)

        # {节点id: [邻居embedding]}
        pos_nodes_nei = {pos_node.item(): pos_feat[pos_g.in_edges(pos_node)] for pos_node in pos_nodes}
        neg_nodes_nei = {neg_node.item(): neg_feat[neg_g.in_edges(neg_node)] for neg_node in neg_nodes}

        neg_loss = []
        # 另一视图的邻居节点为负样本
        for node, feat in zip(pos_nodes, h_pos):
            neg_loss.append(torch.sum(F.cosine_similarity(feat, neg_nodes_nei[node.item()])))
        for node, feat in zip(neg_nodes, h_neg):
            neg_loss.append(torch.sum(F.cosine_similarity(feat, pos_nodes_nei[node.item()])))
        # (2N, 1)
        neg_loss = torch.stack(neg_loss)
        
        total_loss = torch.log(torch.sum(torch.exp(torch.cat([pos_loss, neg_loss], dim=0)))) - torch.sum(pos_loss)
        return total_loss, h_pos

    def forward(self, g):
        '''
        :g 二分图 需要带节点的特征feat
        '''
        predict_type_id = g.ntypes.index(self.predict_type)
        # 扰动
        g_neg = fuse_graph(g, self.fuse_way)

        # 二分图上先进行GCN 同类型了邻居的重要性相同 TODO 可以改为attention
        homo_g = dgl.to_homogeneous(g, ndata=['feat'])
        homo_feat = homo_g.ndata['feat']
        homo_g = dgl.add_self_loop(homo_g)
        h = self.graph_encoder(homo_g, homo_feat)
        # 扰动的图进行GCN
        homo_neg_g = dgl.to_homogeneous(g_neg, ndata=['feat'])
        homo_neg_feat = homo_neg_g.ndata['feat']
        homo_neg_g = dgl.add_self_loop(homo_neg_g)
        neg_g_h = self.graph_encoder(homo_neg_g, homo_neg_feat)

        loss, predict_h = self._calculate_loss(homo_g, h, homo_neg_g, neg_g_h, predict_type_id)
        return loss, predict_h


class Contrast(nn.Module):
    '''
    全图对比学习层
    in_dim: 输入维数
    hidden_dim: 隐藏层维数
    etypes: 图的关系三元组(src, edge_type, dst)
    '''
    def __init__(self, in_dim, hidden_dim, etypes, attn_drop=0.5, tem=0.7):
        super().__init__()
        self.bigraphs = nn.ModuleDict({
            etype: BiGraphContrastLayer(in_dim, hidden_dim, dtype, tem=tem) for _, etype, dtype in etypes
        })
        self.attention = nn.ModuleDict({
            dtype: Attention(hidden_dim, attn_drop) for _, _, dtype in etypes
        })
    
    def forward(self, g, feat):
        '''
        :g 异构图
        :feat 节点特征Dict {ntype: [nfeat]}  
        '''
        e_feat = {}
        n_feat = {}
        local_loss = []
        for stype, etype, dtype in g.canonical_etypes():
            if dtype not in n_feat:
                n_feat[dtype] = []
            # 构造二分图 每种边的关系对应一种二分图 {etype: BiGraph}
            bigraph = dgl.heterograph({etype: g.edges(etype=etype)})
            bigraph.nodes[stype].data['h'] = feat[stype]
            bigraph.nodes[dtype].data['h'] = feat[dtype]
            # bi_feat为二分图目标节点的表示, bi_loss为当前二分图的对比损失
            bi_feat, bi_loss = self.bigraphs[etype](bigraph)
            e_feat[etype] = bi_feat # 记录一下当前关系下的目标节点表示 后面用于跨关系消息传递
            n_feat[dtype].append(bi_feat) # 同目标类型的节点的局部embedding
            local_loss.append(bi_loss) # 目前是直接将所有二分图的loss相加作为局部loss TODO 和节点表示的权重结合
        for dtype in n_feat:
            # [(N, d)] -> (N, M, d)
            h = torch.stack(n_feat[dtype], dim=1)
            z = self.attention[dtype](h) #(N, d)
            n_feat[dtype] = z
            
