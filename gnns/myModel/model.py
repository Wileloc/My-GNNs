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

    def forward(self, stype, etype, dtype, g, feat):
        '''
        :g 二分图
        :feat (tensor(N_src, d_in), tensor(N_dst, d_in)) 输入特征
        '''
        if stype == dtype:
            num_nodes_dict = {stype: g.num_src_nodes(stype)}
        else:
            num_nodes_dict = {stype: g.num_src_nodes(stype), dtype: g.num_dst_nodes(dtype)}
        bigraph = dgl.heterograph({
            (stype, etype, dtype): g.edges(etype=etype)}, 
            num_nodes_dict=num_nodes_dict
        )
        if stype == dtype:
            bigraph.nodes[stype].data['h'] = feat
        else:
            bigraph.nodes[stype].data['h'] = feat
            bigraph.nodes[dtype].data['h'] = torch.rand_like(feat)
        # with g.local_scope():
        homo_g = dgl.to_homogeneous(bigraph, ndata=['h'])
        homo_feat = homo_g.ndata['h']
        homo_g = dgl.add_self_loop(homo_g)
        h = self.graph_encoder(homo_g, homo_feat)

        predict_type_id = bigraph.ntypes.index(self.predict_type)
        pos_nodes = homo_g.filter_nodes(lambda nodes: (nodes.data['_TYPE'] == predict_type_id))
        return h[pos_nodes]


class ContrastLayer(nn.Module):
    '''
    全图对比学习层
    in_dim: 输入维数
    hidden_dim: 隐藏层维数
    etypes: 图的关系三元组(src, edge_type, dst)
    '''
    def __init__(self, in_dim, hidden_dim, etypes):
        super().__init__()
        self.bigraphs = nn.ModuleDict({
            etype: BiGraphContrastLayer(in_dim, hidden_dim, dtype) for _, etype, dtype in etypes
        })
        # self.hidden_dim = hidden_dim
        # self.activation = nn.PReLU()
    
    def forward(self, g, feat):
        '''
        :g 异构图
        :feat 节点特征Dict 每种类型的顶点在不同关系下(dst作为目标节点)的表示{ntype: {etype: Tensor(N, d)}}
        return: 二分图对比学习 + 跨关系学习后的节点表征Dict {ntype: {etype: Tensor(N, d)}}, 二分图对比学习的loss
        '''
        
        n_feat = {
            dtype: {
                etype+"-"+s_etype: self.bigraphs[etype](stype, etype, dtype, g[etype], feat[stype][s_etype])[:g.num_dst_nodes(dtype)] for s_etype in feat[stype]
            } for stype, etype, dtype in g.canonical_etypes if g.num_edges(etype) != 0
        }
        
        # n_feat = {}
        # for stype, etype, dtype in g.canonical_etypes:
        #     if g.num_edges(etype) == 0:
        #         continue
        #     if dtype not in n_feat:
        #         n_feat[dtype] = {}
        #     if stype == dtype:
        #         num_nodes_dict = {stype: g.num_src_nodes(stype)}
        #     else:
        #         num_nodes_dict = {stype: g.num_src_nodes(stype), dtype: g.num_dst_nodes(dtype)}
        #     bigraph = dgl.heterograph({
        #         (stype, etype, dtype): g.edges(etype=etype)}, 
        #         num_nodes_dict=num_nodes_dict
        #     )
        #     # 选择这条关系包含的节点 TODO src节点的特征选择的依据？为什么只选择rev_etype 会不会导致重复学习自己的特征 因为rev_etype在L-1层的来源是dtype类型的节点
        #     for s_etype in feat[stype]:
        #         if stype == dtype:
        #             bigraph.nodes[stype].data['h'] = feat[stype][s_etype]
        #         else:
        #             bigraph.nodes[stype].data['h'] = feat[stype][s_etype]
        #             # bigraph.nodes[dtype].data['h'] = feat[dtype][etype][:torch.max(g.edges(etype=etype)[1])+1]
        #             bigraph.nodes[dtype].data['h'] = torch.rand((g.num_dst_nodes(dtype), self.hidden_dim)).cuda()
        #         # 二分图目标节点的表示, TODO 这里使用哪个bigraph？是etype还是s_etype
        #         n_feat[dtype][etype+"-"+s_etype] = self.activation(self.bigraphs[etype](bigraph)[:g.num_dst_nodes(dtype)])

        return n_feat


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
            [ContrastLayer(hidden_dim, hidden_dim, etypes) for _ in range(layer)
        ])
        self.semantic_attention = nn.ModuleDict({
            ntype: Attention(hidden_dim, attn_drop) for ntype, _ in in_dims.items()
        })
        self.etype_stype = {
            etype: stype for stype, etype, _ in etypes
        }
        self.transformer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=attn_drop)
        # predict_etype_number = sum([e[2] == self.predict_ntype for e in etypes])
        self.predict = nn.Linear(hidden_dim * len(in_dims), out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.predict.weight, gain)
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
            # 初始路径为etype
            n_feat[dtype][etype] = torch.mm(self.fc_in[dtype](feat[dtype]), self.in_weights[etype])

        h = {}
        predict_nodes_num = blocks[-1].num_dst_nodes(self.predict_ntype)
        # h[self.predict_ntype] = [n_feat[self.predict_ntype][]]
        for block, layer in zip(blocks, self.contrast):
            n_feat= layer(block, n_feat)
            for etype in n_feat[self.predict_ntype]:
                begin_ntype = self.etype_stype[etype.split('-')[-1]]
                if begin_ntype not in h:
                    h[begin_ntype] = []
                h[begin_ntype].append(n_feat[self.predict_ntype][etype][:predict_nodes_num])
        
        n_z = [self.semantic_attention[ntype](torch.stack(h[ntype], dim=1)) for ntype in h]
        # n_z.append(feat[self.predict_ntype][:predict_nodes_num])
        # h = torch.stack(h, dim=1)
        tgt = torch.stack(n_z, dim=1)
        z = self.transformer(tgt, tgt)
        # print(len(h)) TODO 考虑怎样聚合？
        z = self.predict(z.reshape(z.size()[0], -1))
        return z
