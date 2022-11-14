import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from gnns.myModel.model_mix import MyModelFull
from gnns.utils import (METRICS_STR, add_node_feat, calc_metrics, get_device,
                        load_data, set_random_seed, hg_metapaths)


def train(args):
    torch.autograd.set_detect_anomaly(True)
    if args.seed is not None:
        set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device, reverse_self=False)
    add_node_feat(g, 'one-hot')
    
    all_mps = hg_metapaths(g.to(torch.device('cpu')), predict_ntype, args.num_layers+1)
    layer_metapaths = {
        l: [] for l in range(args.num_layers)
    }
    for mp in all_mps:
        if len(mp.split('-')) - 2 < 0: continue
        layer_metapaths[len(mp.split('-'))-2].append(mp)

    mpnums = len([mp for mp in all_mps if mp.split('-')[0] == predict_ntype])

    model = MyModelFull(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes}, 
        args.num_hidden, data.num_classes, g.canonical_etypes, args.num_layers, predict_ntype, layer_metapaths, mpnums, attn_drop=args.dropout
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        train_logits = model(g, g.srcdata['feat'])
        train_labels = labels[train_idx]
        loss = F.cross_entropy(train_logits[train_idx], train_labels)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(('Epoch {:d} | Loss {:.4f} | ' + METRICS_STR).format(
            epoch, loss.item(), *evaluate(model, g, labels, train_idx, val_idx, test_idx)
        ))

    if args.save_path:
        torch.save(model.cpu().state_dict(), args.save_path)
        print('模型已保存到', args.save_path)


@torch.no_grad()
def evaluate(model, g, labels, train_idx, val_idx, test_idx):
    """评估模型性能

    :param model: nn.Module GNN模型
    :param loader: NodeDataLoader 图数据加载器
    :param g: DGLGraph 图
    :param labels: tensor(N) 顶点标签
    :param num_classes: int 类别数
    :param predict_ntype: str 目标顶点类型
    :param train_idx: tensor(N_train) 训练集顶点id
    :param val_idx: tensor(N_val) 验证集顶点id
    :param test_idx: tensor(N_test) 测试集顶点id
    :param evaluator: ogb.nodeproppred.Evaluator
    :return: train_acc, val_acc, test_acc, train_f1, val_f1, test_f1
    """
    model.eval()
    logits = model(g, g.srcdata['feat'])
    return calc_metrics(logits, labels, train_idx, val_idx, test_idx)


def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument('--device', type=int, default=3, help='GPU 设备')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')
    parser.add_argument('--dataset', choices=['acm', 'dblp'], default='acm', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='The hidden dim')
    parser.add_argument('--num-layers', type=int, default=4, help='层数')
    parser.add_argument('--pre-layers', type=int, default=2, help='MLP层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=11, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--alpha', type=float, default=0.5, help='对比损失的平衡系数')
    parser.add_argument('--save-path', help='模型保存路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
