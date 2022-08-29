import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv, GATConv
import dgl
from gnns.utils.graph_fuse import fuse_graph
import torch.nn.functional as F
from dgl.utils import expand_as_pair


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
    def __init__(self, in_dim, hidden_dim, predict_type, num_heads=8, attn_drop=0.0, fuse_way='drop-edge', tem=0.7, edge_drop_rate=0.01):
        super().__init__()
        self.predict_type = predict_type
        self.fuse_way = fuse_way
        self.tem = tem
        self.edge_drop_rate=edge_drop_rate

    def _calculate_loss(self, pos_g, neg_g):
        h_pos = pos_g.dstdata['ft']
        h_neg = neg_g.dstdata['ft']

        pos_loss = F.cosine_similarity(h_pos, h_neg)

        # pos_nei_feat = pos_g.srcdata['ft'][pos_g.edges(etype=pos_g.etypes[0])[0]]
        # neg_nei_feat = neg_g.srcdata['ft'][neg_g.edges(etype=neg_g.etypes[0])[0]]
        # neg_ebd_feat = neg_g.srcdata['ft'][pos_g.edges(etype=pos_g.etypes[0])[1]]
        # pos_ebd_feat = pos_g.srcdata['ft'][neg_g.edges(etype=neg_g.etypes[0])[1]]
        # neg_loss = []
        # # 另一视图的邻居节点为负样本
        # neg_loss.append(F.cosine_similarity(pos_nei_feat, neg_ebd_feat))
        # neg_loss.append(F.cosine_similarity(neg_nei_feat, pos_ebd_feat))
        # # (2N, 1)
        # neg_loss = torch.cat(neg_loss, dim=0)

        # torch.log(torch.sum(torch.exp(torch.cat([pos_loss, neg_loss], dim=0)/self.tem)) - torch.sum(torch.exp(pos_loss/self.tem)))

        return torch.log(torch.sum(torch.exp(pos_loss/self.tem)))

    def forward(self, g, feats):
        '''
        :g 二分图
        :feats (tensor(N_src, d_in), tensor(N_dst, d_in)) 输入特征
        '''
        with g.local_scope():
            feat_src, feat_dst = expand_as_pair(feats, g)
            g.srcdata['ft'] = feat_src

            # 扰动
            g_neg = fuse_graph(g, self.fuse_way, self.edge_drop_rate)

            # GCN聚合
            g.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))
            g_neg.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))
            loss = self._calculate_loss(g, g_neg)
            return g.dstdata['ft'], loss
            # TODO GAT聚合


class ContrastLayer(nn.Module):
    '''
    全图对比学习层
    in_dim: 输入维数
    hidden_dim: 隐藏层维数
    etypes: 图的关系三元组(src, edge_type, dst)
    '''
    def __init__(self, in_dim, hidden_dim, etypes, fuse_way='drop-edge', attn_drop=0.5, tem=0.7, edge_drop_rate=0.01):
        super().__init__()
        self.bigraphs = nn.ModuleDict({
            etype: BiGraphContrastLayer(in_dim, hidden_dim, dtype, tem=tem, fuse_way=fuse_way, edge_drop_rate=edge_drop_rate) for _, etype, dtype in etypes
        })
        self.rev_etype = {
            e: next(re for rs, re, rd in etypes if rs == d and rd == s and re != e)
            for s, e, d in etypes
        }
    
    def forward(self, g, feats):
        '''
        :g 异构图
        :feat 节点特征Dict 每种类型的顶点在不同关系下(dst作为目标节点)的表示{(stype, etype, dtype): Tensor(N, d)}
        return: 二分图对比学习 + 跨关系学习后的节点表征Dict {(stype, etype, dtype): Tensor(N, d)}, 二分图对比学习的loss
        '''

        if g.is_block:
            feats_dst = {r: feats[r][:g.num_dst_nodes(r[2])] for r in feats}
        else:
            feats_dst = feats

        n_feat = {}
        local_loss = []
        for stype, etype, dtype in g.canonical_etypes:
            if g.num_edges((stype, etype, dtype)) > 0:
                n_feat[(stype, etype, dtype)], bi_loss = self.bigraphs[etype](
                    g[stype, etype, dtype], 
                    (feats[(dtype, self.rev_etype[etype], stype)], feats_dst[(stype, etype, dtype)])
                )
                local_loss.append(bi_loss)# 目前是直接将所有二分图的loss相加作为局部loss TODO 和节点表示的权重结合
        return n_feat, torch.sum(torch.stack(local_loss))


class MyModel(nn.Module):
    '''
    :in_dim 输入特征维数
    :hidden_dim 隐层特征维数
    :etypes [(stype, etype, dtype)] 边关系三元组
    :layer 卷积层数
    :attn_drop Attention Dropout概率
    '''

    def __init__(self, in_dims, hidden_dim, out_dim, etypes, layer, predict_ntype, fuse_way='drop-edge', attn_drop=0.5, tem=0.7, edge_drop_rate=0.01) -> None:
        super().__init__()
        self.etypes = etypes
        self.predict_ntype = predict_ntype
        self.fc_in = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()
        })
        self.contrast = nn.ModuleList(
            [ContrastLayer(hidden_dim, hidden_dim, etypes, fuse_way=fuse_way, tem=tem, edge_drop_rate=edge_drop_rate) for _ in range(layer)
        ])
        self.attention = nn.ModuleDict({
            etype: Attention(hidden_dim, attn_drop) for _, etype, _ in etypes
        })
        self.attn = Attention(hidden_dim, attn_drop)
        self.predict = nn.Linear(hidden_dim, out_dim)
        self.co_loss = None
        # self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.predict.weight, gain)
        for ntype in self.fc_in:
            nn.init.xavier_normal_(self.fc_in[ntype].weight, gain)

    def get_co_loss(self):
        return self.co_loss

    def forward(self, blocks, feat):
        '''
        :g DGLGraph 异构图
        :feat 节点特征Dict {ntype: Tensor(N, d_i)}
        return 节点表征Dict {ntype: Tensor(N, d_h)}
        '''
        n_feat = {
            (stype, etype, dtype): self.fc_in[dtype](feat[dtype])
            for stype, etype, dtype in self.etypes
        }
        
        # L层
        for block, layer in zip(blocks, self.contrast):
            n_feat, self.co_loss = layer(block, n_feat) # TODO 这里二分图对比的局部损失没有用上

        cross_feat = {
            (e, d): self.attention[e](torch.stack([n_feat[(s, e, d)]], dim=1)) for s, e, d in n_feat
        }

        # TODO 最后每种关系的节点表征如何结合成统一的节点表示
        # 先用语义层次的attention试一下
        z = {
            d: self.predict(self.attn(torch.stack([cross_feat[(e, d)]], dim=1))) for e, d in cross_feat
        }
        return z[self.predict_ntype]
