import dgl
import numpy as np
import torch
import dgl.function as fn

# 定义同构图、异构图
g = dgl.graph(([0,0,0,0,0], [1,2,3,4,5]))
print(g)

g1 = dgl.heterograph({
    ('p', 'cites', 'p'): ([0,0,0], [1,2,3]),
    ('a', 'writes', 'p'): ([0, 0, 1, 1], [0, 1, 2, 3]),
    ('p', 'belongs', 'f'): ([0,1,2,3], [0,1,2,3])
})

print(g1.edges(etype='cites'))

# 进行消息传递
g.ndata['x'] = torch.ones(6,2)
g.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'h'))
print(g.ndata['h'])
