import torch
import dgl


def fuse_graph(g, fuse_way='drop-edge'):
    if fuse_way == 'drop-edge':
        fuse_g = _drop_graph_edge(g)
    else:
        raise ValueError(f'fuse graph: unknown fuse way')
    return fuse_g


def _drop_graph_edge(g, drop_rate=0.01):
    fuse_g = g.cpu() # copy g
    fuse_g = fuse_g.to(g.device)
    drop_edge = torch.squeeze(torch.randint(fuse_g.num_edges(), (1, int(drop_rate*fuse_g.num_edges()))))
    fuse_g.remove_edges(drop_edge.cuda())
    return fuse_g
