import torch
import dgl


def fuse_graph(g, fuse_way='drop-edge', drop_rate=0.01):
    if fuse_way == 'drop-edge':
        fuse_g = _drop_graph_edge(g, drop_rate)
    elif fuse_way == 'attr-mask':
        fuse_g = _attribute_masking(g, drop_rate)
    else:
        raise ValueError(f'fuse graph: unknown fuse way')
    return fuse_g


def _drop_graph_edge(g, drop_rate):
    fuse_g = g.cpu() # copy g
    fuse_g = fuse_g.to(g.device)
    drop_edge = torch.squeeze(torch.randint(fuse_g.num_edges(), (1, int(drop_rate*fuse_g.num_edges()))))
    fuse_g.remove_edges(drop_edge.cuda())
    return fuse_g


def _attribute_masking(g, masking_rate):
    fuse_g = g.cpu()
    fuse_g = fuse_g.to(g.device)
    feat_dict = fuse_g.ndata['h']
    for ntype in feat_dict:
        mask_ntype = torch.rand_like(feat_dict[ntype]).cuda() > masking_rate
        feat_dict[ntype] = feat_dict[ntype] * mask_ntype
    fuse_g.ndata['h'] = feat_dict
    return fuse_g
