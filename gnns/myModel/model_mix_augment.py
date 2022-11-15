import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv
import dgl


class BiGraphContrastLayer(nn.Module):

    '''
    二分图对比学习层
    ----
    参数
    :in_dim 输入特征维数
    :hidden_dim 图编码器特征输出维数 即隐藏层维数
    :predict_type 对比学习的顶点类型 边的目标顶点
    '''
    def __init__(self, in_dim, hidden_dim, predict_type, attn_drop=0.5):
        super().__init__()
        self.predict_type = predict_type
        num_heads = 8
        self.graph_encoder = GATConv(in_dim, hidden_dim // num_heads, num_heads, attn_drop=attn_drop)
    
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
            h = h.flatten(start_dim=1)

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
    def __init__(self, in_dim, hidden_dim, metapaths, predict_ntype, attn_drop):
        super().__init__()
        self.bigraphs = nn.ModuleDict({
            mp: BiGraphContrastLayer(in_dim, hidden_dim, mp.split('-')[0], attn_drop) for mp in metapaths
        })
        self.predict_ntype = predict_ntype
    
    def forward(self, g, feat, layer_idx, max_hops):
        '''
        :g 异构图
        :feat 节点特征Dict 每种类型的顶点在不同关系下(dst作为目标节点)的表示{ntype: {etype: Tensor(N, d)}}
        return: 二分图对比学习 + 跨关系学习后的节点表征Dict {ntype: {etype: Tensor(N, d)}}, 二分图对比学习的loss
        '''

        n_feat = {}
        for stype, etype, dtype in g.canonical_etypes:
            if g.num_edges(etype) == 0:
                continue
            if layer_idx == max_hops - 1 and dtype != self.predict_ntype:
                continue
            if dtype not in n_feat:
                n_feat[dtype] = {dtype: feat[dtype][dtype][:g.num_dst_nodes(dtype)]}
            if stype == dtype:
                num_nodes_dict = {stype: g.num_src_nodes(stype)}
            else:
                num_nodes_dict = {stype: g.num_src_nodes(stype), dtype: g.num_dst_nodes(dtype)}
            bigraph = dgl.heterograph({
                (stype, etype, dtype): g.edges(etype=etype)}, 
                num_nodes_dict=num_nodes_dict
            )
            for s_etype in feat[stype]:
                if len(s_etype.split('-')) - 1 != layer_idx: continue
                if stype == dtype:
                    bigraph.nodes[stype].data['h'] = feat[stype][s_etype]
                else:
                    bigraph.nodes[stype].data['h'] = feat[stype][s_etype]
                    bigraph.nodes[dtype].data['h'] = feat[dtype][dtype][:g.num_dst_nodes(dtype)]
                # 二分图目标节点的表示
                n_feat[dtype][dtype+"-"+s_etype] = self.bigraphs[dtype+"-"+s_etype](bigraph)[:g.num_dst_nodes(dtype)]

        return n_feat


class MyModel(nn.Module):
    '''
    :in_dim 输入特征维数
    :hidden_dim 隐层特征维数
    :etypes [(stype, etype, dtype)] 边关系三元组
    :layer 卷积层数
    :attn_drop Attention Dropout概率
    '''

    def __init__(self, in_dims, hidden_dim, out_dim, etypes, layer, predict_ntype, layer_metapaths, metapath_numbers, attn_drop=0.5) -> None:
        super().__init__()
        self.predict_ntype = predict_ntype
        self.fc_in = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()
        })
        self.contrast = nn.ModuleList(
            [ContrastLayer(hidden_dim, hidden_dim, mps, predict_ntype, attn_drop) for _, mps in layer_metapaths.items()
        ])
        self.transformer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=attn_drop)
        self.predict = nn.Linear(hidden_dim * metapath_numbers, out_dim)
        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.predict.weight, gain)
        for ntype in self.fc_in:
            nn.init.xavier_normal_(self.fc_in[ntype].weight, gain)

    def forward(self, blocks, feat, labels=None):
        '''
        :g DGLGraph 异构图
        :feat 节点特征Dict {ntype: Tensor(N, d_i)}
        return 节点表征Dict {ntype: Tensor(N, d_h)}
        '''
        n_feat = {
            dtype: {dtype: self.fc_in[dtype](feat[dtype])} for dtype in feat
        }

        predict_nodes_num = blocks[-1].num_dst_nodes(self.predict_ntype)
        h = {
            self.predict_ntype: n_feat[self.predict_ntype][self.predict_ntype][:predict_nodes_num]
        }
        for idx, (block, layer) in enumerate(zip(blocks, self.contrast)):
            n_feat = layer(block, n_feat, idx, len(self.contrast))
            for dtype in n_feat:
                if dtype == self.predict_ntype: continue
                remove = []
                for path in n_feat[dtype]:
                    if len(path.split('-')) - 2 < idx and len(path.split('-')) - 2 >= 0:
                        remove.append(path)
                for path in remove:
                    n_feat[dtype].pop(path)
            for path in n_feat[self.predict_ntype]:
                if len(path.split('-')) - 2 == idx:
                    h[path] = n_feat[self.predict_ntype][path][:predict_nodes_num]

        tgt = torch.stack(list(h.values()), dim=1)
        z = self.transformer(tgt, tgt)
        if labels is not None:
            augment_labels = torch.unique(labels)
            augment_z = torch.stack([torch.mean(z[labels == l], dim=0) for l in augment_labels])
            z = torch.cat((z, augment_z))
            z = self.predict(z.reshape(z.size()[0], -1))
            return z, augment_labels
        z = self.predict(z.reshape(z.size()[0], -1))
        return z


class MyModelFull(MyModel):

    def forward(self, g, feat):
        return super().forward([g] * len(self.contrast), feat)