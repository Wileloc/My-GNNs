import torch
import torch.nn as nn


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


class MyModel(nn.Module):
    '''
    :in_dim 输入特征维数
    :hidden_dim 隐层特征维数
    :etypes [(stype, etype, dtype)] 边关系三元组
    :layer 卷积层数
    :attn_drop Attention Dropout概率
    '''

    def __init__(self, in_dims, hidden_dim, out_dim, layer, predict_ntype, ntype_nums, attn_drop=0.5) -> None:
        super().__init__()
        self.predict_ntype = predict_ntype
        self.fc_in = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()
        })
        self.in_dropout = nn.Dropout(attn_drop)
        self.res_fc = nn.Linear(in_dims[predict_ntype], hidden_dim)
        self.transformer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=attn_drop)
        self.ntype_projection = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim * num, hidden_dim) for ntype, num in ntype_nums.items()
        })
        # self.attention = nn.ModuleDict({
        #     ntype: Attention(hidden_dim, attn_drop) for ntype in in_dims
        # })
        # self.node_attention = Attention(hidden_dim, attn_drop)
        length = hidden_dim * (len(in_dims) + 1)
        # length = hidden_dim
        predict_layer = [
            [nn.Linear(length, length), nn.PReLU(), nn.Dropout(attn_drop)] 
            for _ in range(layer)
        ]
        self.predict = nn.Sequential(*(
            [ele for lr in predict_layer for ele in lr] + [nn.Linear(length, out_dim)]
        ))
        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for ntype in self.fc_in:
            nn.init.xavier_normal_(self.fc_in[ntype].weight, gain)
        nn.init.xavier_normal_(self.res_fc.weight, gain)
        for layer in self.predict:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain)
                nn.init.zeros_(layer.bias)
        # for ntype in self.ntype_projection:
        #     nn.init.xavier_normal_(self.ntype_projection[ntype].weight, gain)

    def forward(self, feat):
        '''
        :g DGLGraph 异构图
        :feat 节点特征Dict {ntype: Tensor(N, d_i)}
        :return 节点表征Dict {ntype: Tensor(N, d_h)}
        '''
        tgt_feat = feat[self.predict_ntype]
        feat = {
            path: self.fc_in[path.split('-')[-1]](self.in_dropout(feat[path])) for path in feat
        }

        # x = torch.stack(list(feat.values()), dim=1)
        ntype_x = {}
        for path in feat:
            ntype = path.split('-')[-1]
            if ntype not in ntype_x:
                ntype_x[ntype] = []
            ntype_x[ntype].append(feat[path])
        # ntype_x = {
        #     ntype: torch.stack(ntype_x[ntype], dim=1) for ntype in ntype_x if len(ntype_x[ntype]) != 0
        # }
        ntype_x = {
            ntype: self.ntype_projection[ntype](torch.cat(ntype_x[ntype], dim=1)) for ntype in ntype_x if len(ntype_x[ntype]) != 0
        }
        # ntype_x = {
        #     ntype: self.transformer(ntype_x[ntype], x) for ntype in ntype_x
        # }
        z = list(ntype_x.values())
        z.append(self.res_fc(tgt_feat))
        x = torch.stack(z, dim=1)
        x = self.transformer(x, x)
        x = torch.reshape(x, (x.size()[0], -1))
        # z = [self.attention[ntype](ntype_x[ntype]) for ntype in ntype_x]

        # z.append(self.res_fc(tgt_feat))
        x = self.predict(x)
        # TODO dblp效果很好 预测顶点时author，但acm等paper分类任务上不好
        # z = self.node_attention(torch.stack(z, dim=1))
        # z = z + self.res_fc(tgt_feat)
        # z = self.predict(z)
        return x
