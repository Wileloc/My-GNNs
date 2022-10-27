import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv
import dgl


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
    '''
    def __init__(self, in_dim, hidden_dim, predict_type, norm='right', activation=nn.PReLU()):
        super().__init__()
        self.predict_type = predict_type
        # TODO norm方式之前是right 试一下both和left
        self.graph_encoder = GraphConv(in_dim, hidden_dim, norm=norm, activation=activation)
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.graph_encoder.weight, gain)
        nn.init.xavier_normal_(self.graph_encoder.bias, gain)

    def forward(self, g):
        '''
        :g 二分图
        :feat (tensor(N_src, d_in), tensor(N_dst, d_in)) 输入特征
        '''
        with g.local_scope():
            homo_g = dgl.to_homogeneous(g, ndata=['h'])
            homo_feat = homo_g.ndata['h']
            homo_g = dgl.add_self_loop(homo_g)
            h = self.graph_encoder(homo_g, homo_feat)

            if len(g.ntypes) == 1:
                predict_node_id = torch.max(g.edges(etype=g.etypes[0])[1])
                pos_nodes = homo_g.filter_nodes(lambda nodes: (nodes.data['_ID'] <= predict_node_id))
            else:
                predict_type_id = g.ntypes.index(self.predict_type)
                pos_nodes = homo_g.filter_nodes(lambda nodes: (nodes.data['_TYPE'] == predict_type_id))
            return h[pos_nodes]


class ContrastLayer(nn.Module):
    '''
    全图对比学习层
    in_dim: 输入维数
    hidden_dim: 隐藏层维数
    etypes: 图的关系三元组(src, edge_type, dst)
    '''
    def __init__(self, in_dim, hidden_dim, etypes, attn_drop=0.5):
        super().__init__()
        self.bigraphs = nn.ModuleDict({
            etype: BiGraphContrastLayer(in_dim, hidden_dim, dtype) for _, etype, dtype in etypes
        })
        self.transformer = nn.ModuleDict({
            etype: nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=attn_drop) for _, etype, _ in etypes
        })
        # self.rev_etype = {
        #     e: next(re for rs, re, rd in etypes if rs == d and rd == s and re != e)
        #     for s, e, d in etypes
        # }
        # self.activation = nn.ReLU()
    
    def forward(self, g, feat):
        '''
        :g 异构图
        :feat 节点特征Dict 每种类型的顶点在不同关系下(dst作为目标节点)的表示{ntype: {etype: Tensor(N, d)}}
        return: 二分图对比学习 + 跨关系学习后的节点表征Dict {ntype: {etype: Tensor(N, d)}}, 二分图对比学习的loss
        '''
        n_feat = {}
        dtype_feat = {}
        for dtype in feat:
            for etype in feat[dtype]:
                if dtype not in dtype_feat:
                    dtype_feat[dtype] = torch.zeros_like(feat[dtype][etype])
                dtype_feat[dtype] = torch.add(dtype_feat[dtype], feat[dtype][etype])
        for stype, etype, dtype in g.canonical_etypes:
            if g.num_edges(etype) == 0:
                continue
            if dtype not in n_feat:
                n_feat[dtype] = {}
            # 构造二分图 每种边的关系对应一种二分图 {etype: BiGraph}
            # 二分图的输入特征为, dst节点在当前关系下的表征与src节点在当前关系的reverse关系下的表征
            bigraph = dgl.heterograph({(stype, etype, dtype): g.edges(etype=etype)})
            # 选择这条关系包含的节点 TODO src节点的特征选择的依据？为什么只选择rev_etype 会不会导致重复学习自己的特征 因为rev_etype在L-1层的来源是dtype类型的节点
            if stype == dtype:
                bigraph.nodes[stype].data['h'] = dtype_feat[stype][:torch.max(torch.max(g.edges(etype=etype)[0])+1, torch.max(g.edges(etype=etype)[1])+1)]
            else:
                bigraph.nodes[stype].data['h'] = dtype_feat[stype][:torch.max(g.edges(etype=etype)[0])+1]
                bigraph.nodes[dtype].data['h'] = feat[dtype][etype][:torch.max(g.edges(etype=etype)[1])+1]
            # 二分图目标节点的表示
            n_feat[dtype][etype] = self.bigraphs[etype](bigraph) # 同目标类型的节点的局部embedding, 同时记录边关系类型, 用于跨关系消息传递

        cross_feat = {}
        for dtype in n_feat:
            cross_feat[dtype] = {}
            for etype in n_feat[dtype]:
                nodes_id = torch.max(g.edges(etype=etype)[1])+1
                cross_feat[dtype][etype] = feat[dtype][etype][:g.num_dst_nodes(dtype)]
                cross_feat[dtype][etype][:nodes_id] = n_feat[dtype][etype]

        d_h = {
            dtype: [cross_feat[dtype][e] for e in cross_feat[dtype]] for dtype in cross_feat
        }

        for dtype in cross_feat:
            if len(cross_feat[dtype]) == 1:
                continue
            for etype in cross_feat[dtype]:
                h = torch.stack(d_h[dtype], dim=1) # (N, M, d)
                # z = self.attention[etype](h) #(N, d)
                # TODO 如果前面src节点特征改了的话这里也要改
                tgt = torch.unsqueeze(feat[dtype][etype][:g.num_dst_nodes(dtype)], dim=1)
                # cross_feat[dtype][etype] = feat[dtype][etype][:g.num_dst_nodes(dtype)] + z
                cross_feat[dtype][etype] = torch.squeeze(self.transformer[etype](tgt, h), dim=1)

        return cross_feat


class MyModel(nn.Module):
    '''
    :in_dim 输入特征维数
    :hidden_dim 隐层特征维数
    :etypes [(stype, etype, dtype)] 边关系三元组
    :layer 卷积层数
    :attn_drop Attention Dropout概率
    '''

    def __init__(self, in_dims, hidden_dim, out_dim, etypes, layer, predict_ntype, attn_drop=0.5) -> None:
        super().__init__()
        self.etypes = etypes
        self.predict_ntype = predict_ntype
        self.fc_in = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()
        })
        self.in_weights = nn.ParameterDict({
            etype[1]: nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim)) for etype in etypes
        })
        self.contrast = nn.ModuleList(
            [ContrastLayer(hidden_dim, hidden_dim, etypes, attn_drop=attn_drop) for _ in range(layer)
        ])
        # self.attn = Attention(hidden_dim, attn_drop)
        predict_etype_number = sum([e[2] == self.predict_ntype for e in etypes])
        # self.fc_last = nn.Linear(in_dims[self.predict_ntype], hidden_dim)
        # self.hop_attn = nn.ModuleDict({
        #     e[1]: Attention(hidden_dim, attn_drop) for e in etypes if e[2] == self.predict_ntype
        # })
        self.predict = nn.Linear(hidden_dim * predict_etype_number, out_dim)
        # self.predict = nn.Linear(hidden_dim * predict_etype_number * layer, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.predict.weight, gain)
        # nn.init.xavier_normal_(self.fc_last.weight, gain)
        for ntype in self.fc_in:
            nn.init.xavier_normal_(self.fc_in[ntype].weight, gain)
        for etype in self.in_weights:
            nn.init.xavier_normal_(self.in_weights[etype], gain)

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
            n_feat[dtype][etype] = torch.mm(self.fc_in[dtype](feat[dtype]), self.in_weights[etype])
            # n_feat[dtype][etype] = self.fc_in[dtype](feat[dtype])

        for block, layer in zip(blocks, self.contrast):
            n_feat= layer(block, n_feat)

        # TODO 最后每种关系的节点表征如何结合成统一的节点表示
        # 先用语义层次的attention试一下
        # h = torch.stack([n_feat[self.predict_ntype][etype] for etype in n_feat[self.predict_ntype]], dim=1)
        # z = self.predict(self.attn(h))
        h = [n_feat[self.predict_ntype][etype] for etype in n_feat[self.predict_ntype]]
        # h.append(self.fc_last(blocks[-1].dstdata['feat'][self.predict_ntype]))
        z = self.predict(torch.cat(h, dim=1))
        return z

        # hops_feat = []
        # for l in range(1, len(self.contrast)+1):
        #     neighbor_feat = {}
        #     for _, etype, dtype in self.etypes:
        #         if dtype not in neighbor_feat:
        #             neighbor_feat[dtype] = {}
        #         neighbor_feat[dtype][etype] = self.fc_in[dtype](feat[dtype])
        #     for block, layer in zip(blocks[-l:], self.contrast[-l:]):
        #         neighbor_feat = layer(block, neighbor_feat)
        #     hops_feat.append(neighbor_feat[self.predict_ntype])
        
        # hops_feat = [self.hop_attn[e](torch.stack([h[e] for h in hops_feat], dim=1)) for e in hops_feat[0]]
        # hops_feat.append(self.fc_last(blocks[-1].dstdata['feat'][self.predict_ntype]))
        # h = torch.cat(hops_feat, dim=1)
        # z = self.predict(h)
        # return z
