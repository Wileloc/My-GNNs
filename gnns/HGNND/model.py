import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv


class HGNND(nn.Module):

    def __init__(self, ntypes, in_dim, hidden_dim, out_dim, predict_type, k_max) -> None:
        super().__init__()
        self.fc_in = nn.ModuleDict({
            ntype: nn.Linear(n_in_dim, in_dim) for ntype, n_in_dim in ntypes.items()
        })
        self.gat = GATConv(in_dim, hidden_dim, 1, activation=F.elu)
        self.hold = 0.0
        self.k_T = k_max
        self.predict = nn.Linear(hidden_dim, out_dim)
        self.predict_type = predict_type
    
    def _calculate_loss(self, score):
        loss = F.logsigmoid(score).view(-1)
        self.hold = self.k_T * torch.max(loss)
        zeros = torch.zeros_like(loss)
        loss = torch.where(loss > self.hold, zeros, loss)
        return -loss.sum()
    
    def forward(self, g, feat, pos_src, pos_dst, neg_src, neg_dst, alpha_T):
        '''
        g: 图
        feat: 顶点类型到顶点输入特征的映射 Dict[str, tensor(N_i, d_in_i)]
        pos_src: 正例的起始节点
        pos_dst: 正例的终止节点
        (neg_src, neg_dst): 负例
        '''
        feat = {ntype: self.fc_in[ntype](feat[ntype]) for ntype in feat}
        for n in g.ntypes:
            g.nodes[n].data['feat'] = feat[n]
        homo_g = dgl.to_homogeneous(g, ndata=['feat'])

        homo_feat = homo_g.ndata['feat']
        homo_g = dgl.add_self_loop(homo_g)
        h = self.gat(homo_g, homo_feat)

        pos_score = torch.sum((h[pos_src] * h[pos_dst]), dim=1) # (N, d) * (N, d) -> (N, d) -> (N)
        neg_score = torch.sum((h[neg_src] * h[neg_dst]), dim=1)

        if self.k_T > alpha_T:
            self.k_T = alpha_T  # k_T = min(k_T, alpha_T)
        pos_loss = self._calculate_loss(pos_score)
        neg_loss = self._calculate_loss(neg_score)

        predict_type_id = g.ntypes.index(self.predict_type)
        out = self.predict(h[homo_g.filter_nodes(lambda nodes: (nodes.data['_TYPE'] == predict_type_id))])

        return pos_loss + neg_loss, out
