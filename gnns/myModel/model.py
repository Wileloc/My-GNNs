import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv, GATConv
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
    def __init__(self, in_dim, hidden_dim, predict_type, num_heads=8, attn_drop=0.0, fuse_way='drop-edge', tem=0.7, edge_drop_rate=0.01):
        super().__init__()
        self.predict_type = predict_type
        # self.graph_encoder = GraphConv(in_dim, hidden_dim, norm='right', activation=nn.PReLU())
        self.gat_encoder = GATConv(in_dim, hidden_dim, num_heads, attn_drop=attn_drop, activation=nn.PReLU())
        self.fuse_way = fuse_way
        self.tem = tem
        self.edge_drop_rate=edge_drop_rate
        # self.reset_parameters()
    
    def reset_parameters(self):
        self.gat_encoder.reset_parameters()
        # self.graph_encoder.reset_parameters()

    def _calculate_loss(self, pos_g, pos_feat, neg_g, neg_feat, predict_type_id=None, predict_node_id=None):
        if predict_type_id is not None:
            pos_nodes = pos_g.filter_nodes(lambda nodes: (nodes.data['_TYPE'] == predict_type_id))
            neg_nodes = neg_g.filter_nodes(lambda nodes: (nodes.data['_TYPE'] == predict_type_id))
        else:
            pos_nodes = pos_g.filter_nodes(lambda nodes: (nodes.data['_ID'] <= predict_node_id))
            neg_nodes = neg_g.filter_nodes(lambda nodes: (nodes.data['_ID'] <= predict_node_id))
        # (N, hid_d)
        h_pos = pos_feat[pos_nodes]
        # (N, hid_d)
        h_neg = neg_feat[neg_nodes]

        # (N, 1)
        pos_loss = F.cosine_similarity(h_pos, h_neg)

        # {节点id: [邻居embedding]}
        pos_nei_feat = pos_feat[pos_g.in_edges(pos_nodes)[0]]
        neg_nei_feat = neg_feat[neg_g.in_edges(neg_nodes)[0]]
        neg_ebd_feat = neg_feat[pos_g.in_edges(pos_nodes)[1]]
        pos_ebd_feat = pos_feat[neg_g.in_edges(neg_nodes)[1]]

        neg_loss = []
        # 另一视图的邻居节点为负样本
        neg_loss.append(F.cosine_similarity(pos_nei_feat, neg_ebd_feat))
        neg_loss.append(F.cosine_similarity(neg_nei_feat, pos_ebd_feat))
        # (2N, 1)
        neg_loss = torch.cat(neg_loss, dim=0)
        
        # 如果不考虑负样本，只考虑正样本？
        total_loss = torch.log(torch.sum(torch.exp(pos_loss/self.tem)))
        # total_loss = torch.log(torch.sum(torch.exp(torch.cat([pos_loss, neg_loss], dim=0)/self.tem)) - torch.sum(torch.exp(pos_loss/self.tem)))
        return total_loss, h_pos

    def forward(self, g):
        '''
        :g 二分图
        :feat (tensor(N_src, d_in), tensor(N_dst, d_in)) 输入特征
        '''
        with g.local_scope():
            # feat_src, feat_dst = expand_as_pair(feat, g)
            # g.srcdata['h'] = feat_src
            # g.dstdata['h'] = feat_dst
            # 扰动
            g_neg = fuse_graph(g, self.fuse_way, self.edge_drop_rate)
    
            # 二分图上先进行GCN 同类型的邻居的重要性相同 TODO 可以改为attention
            homo_g = dgl.to_homogeneous(g, ndata=['h'])
            homo_feat = homo_g.ndata['h']
            homo_g = dgl.add_self_loop(homo_g)
            # h = self.graph_encoder(homo_g, homo_feat)
            h = self.gat_encoder(homo_g, homo_feat) # （N, H, d_out)
            # 这里先用了mean的方式进行聚合, 与dgl的GATConv的实现有关
            h = torch.mean(h, dim=1)
            # 扰动的图进行GCN
            homo_neg_g = dgl.to_homogeneous(g_neg, ndata=['h'])
            homo_neg_feat = homo_neg_g.ndata['h']
            homo_neg_g = dgl.add_self_loop(homo_neg_g)
            # neg_g_h = self.graph_encoder(homo_neg_g, homo_neg_feat)
            neg_g_h = self.gat_encoder(homo_neg_g, homo_neg_feat)
            neg_g_h = torch.mean(neg_g_h, dim=1)

            if len(g.ntypes) == 1:
                predict_node_id = torch.max(g.edges(etype=g.etypes[0])[1])
                loss, predict_h = self._calculate_loss(homo_g, h, homo_neg_g, neg_g_h, predict_node_id=predict_node_id)
            else:
                predict_type_id = g.ntypes.index(self.predict_type)
                loss, predict_h = self._calculate_loss(homo_g, h, homo_neg_g, neg_g_h, predict_type_id=predict_type_id)
            return loss, predict_h


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
        # self.attention = nn.ModuleDict({
        #     etype: Attention(hidden_dim, attn_drop) for _, etype, _ in etypes
        # })
        self.rev_etype = {
            e: next(re for rs, re, rd in etypes if rs == d and rd == s and re != e)
            for s, e, d in etypes
        }
    
    def forward(self, g, feat):
        '''
        :g 异构图"
        :feat 节点特征Dict 每种类型的顶点在不同关系下(dst作为目标节点)的表示{ntype: {etype: Tensor(N, d)}}
        return: 二分图对比学习 + 跨关系学习后的节点表征Dict {ntype: {etype: Tensor(N, d)}}, 二分图对比学习的loss
        '''
        # e_feat = {}
        n_feat = {}
        local_loss = []
        for stype, etype, dtype in g.canonical_etypes:
            if g.num_edges(etype) == 0:
                continue
            if dtype not in n_feat:
                n_feat[dtype] = {}
            # 构造二分图 每种边的关系对应一种二分图 {etype: BiGraph}
            # 二分图的输入特征为, dst节点在当前关系下的表征与src节点在当前关系的reverse关系下的表征
            bigraph = dgl.heterograph({(stype, etype, dtype): g.edges(etype=etype)})
            # 选择这条关系包含的节点
            if stype == dtype:
                bigraph.nodes[stype].data['h'] = feat[stype][self.rev_etype[etype]][:torch.max(torch.max(g.edges(etype=etype)[0])+1, torch.max(g.edges(etype=etype)[1])+1)]
            else:
                bigraph.nodes[stype].data['h'] = feat[stype][self.rev_etype[etype]][:torch.max(g.edges(etype=etype)[0])+1]
                bigraph.nodes[dtype].data['h'] = feat[dtype][etype][:torch.max(g.edges(etype=etype)[1])+1]
            # bi_feat为二分图目标节点的表示, bi_loss为当前二分图的对比损失
            bi_loss, bi_feat = self.bigraphs[etype](bigraph)
            # e_feat[etype] = bi_feat # 记录一下当前关系下的目标节点表示 后面用于跨关系消息传递
            n_feat[dtype][etype] = bi_feat # 同目标类型的节点的局部embedding, 同时记录边关系类型, 用于跨关系消息传递
            local_loss.append(bi_loss) # 目前是直接将所有二分图的loss相加作为局部loss TODO 和节点表示的权重结合

        # R-HGNN的做法是对每种关系的节点表示, 为每种关系设置一个注意力向量, 聚合其他关系下的节点表示，传入下一层。
        # cross_feat = {}
        # 问题 不能这样做聚合 因为每个顶点关联的关系是不一样的 所以不能直接用矩阵
        for dtype in n_feat:
            for etype in n_feat[dtype]:
                nodes_num = g.num_dst_nodes(dtype)
                if len(n_feat[dtype][etype]) < nodes_num:
                    n_feat[dtype][etype] = torch.cat([n_feat[dtype][etype], torch.randn((nodes_num-len(n_feat[dtype][etype]),64)).cuda()])
                elif len(n_feat[dtype][etype]) > nodes_num:
                    n_feat[dtype][etype] = n_feat[dtype][etype][:nodes_num]
        # for dtype in n_feat:
        #     cross_feat[dtype] = {}
        #     if len(n_feat[dtype]) == 1:
        #         cross_feat[dtype] = n_feat[dtype]
        #         continue
        #     for etype in n_feat[dtype]:
        #         h = torch.stack([n_feat[dtype][e] for e in n_feat[dtype]], dim=1)
        #         z = self.attention[etype](h) #(N, d)
        #         cross_feat[dtype][etype] = z
        # return cross_feat, torch.sum(torch.stack(local_loss))
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
        n_feat = {}
        for _, etype, dtype in self.etypes:
            if dtype not in n_feat:
                n_feat[dtype] = {}
            n_feat[dtype][etype] = self.fc_in[dtype](feat[dtype]) # 初始每个关系下的节点特征都相同 TODO dtype 会不会不一定在这次采样的字图中？
        
        # L层
        for block, layer in zip(blocks, self.contrast):
            n_feat, self.co_loss = layer(block, n_feat) # TODO 这里二分图对比的局部损失没有用上
        cross_feat = {}
        for dtype in n_feat:
            cross_feat[dtype] = {}
            if len(n_feat[dtype]) == 1:
                cross_feat[dtype] = n_feat[dtype]
                continue
            for etype in n_feat[dtype]:
                h = torch.stack([n_feat[dtype][e] for e in n_feat[dtype]], dim=1)
                z = self.attention[etype](h) #(N, d)
                cross_feat[dtype][etype] = z
        # TODO 最后每种关系的节点表征如何结合成统一的节点表示
        # 先用语义层次的attention试一下
        z = {}
        for dtype in n_feat:
            z[dtype] = self.predict(self.attn(torch.stack([cross_feat[dtype][etype] for etype in cross_feat[dtype]], dim=1)))
        return z[self.predict_ntype]
