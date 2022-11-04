import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim

from gnns.data import DBLPDataset
from gnns.data.acm import ACMDataset
from gnns.han.model import HAN
from gnns.utils import set_random_seed, micro_macro_f1_score

DATASET = {
    'acm': ''
}

HETERO_DATASET = {
    'acm': ACMDataset,
    'dblp': DBLPDataset
}


def train(args):
    set_random_seed(args.seed)
    if args.hetero:
        data = HETERO_DATASET[args.dataset]()
        g = data[0]
        gs = [dgl.metapath_reachable_graph(g, metapath) for metapath in data.metapaths]
        for i in range(len(gs)):
            gs[i] = dgl.add_self_loop(dgl.remove_self_loop(gs[i]))
        ntype = data.predict_ntype
        num_classes = data.num_classes
        features = g.nodes[ntype].data['feat']
        labels = g.nodes[ntype].data['label']
        train_mask = g.nodes[ntype].data['train_mask']
        val_mask = g.nodes[ntype].data['val_mask']
        test_mask = g.nodes[ntype].data['test_mask']
    else:
        data = DATASET[args.dataset]()
        gs = data[0]
        num_classes = data.num_classes
        features = gs[0].ndata['feat']
        labels = gs[0].ndata['label']
        train_mask = gs[0].ndata['train_mask']
        val_mask = gs[0].ndata['val_mask']
        test_mask = gs[0].ndata['test_mask']

    model = HAN(
        len(gs), features.shape[1], args.num_hidden, num_classes, args.num_heads, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    score = micro_macro_f1_score
    if args.task == 'clf':
        metrics = 'Epoch {:d} | Train Loss {:.4f} | Train Micro-F1 {:.4f} | Train Macro-F1 {:.4f}' \
                  ' | Val Micro-F1 {:.4f} | Val Macro-F1 {:.4f}'
    else:
        metrics = 'Epoch {:d} | Train Loss {:.4f} | Train NMI {:.4f} | Train ARI {:.4f}' \
                  ' | Val NMI {:.4f} | Val ARI {:.4f}'
    for epoch in range(args.epochs):
        model.train()
        logits = model(gs, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_metrics = score(logits[train_mask], labels[train_mask])
        val_metrics = score(logits[val_mask], labels[val_mask])
        print(metrics.format(epoch, loss.item(), *train_metrics, *val_metrics))

    test_metrics = evaluate(model, gs, features, labels, test_mask, score)
    if args.task == 'clf':
        print('Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(*test_metrics))
    else:
        print('Test NMI {:.4f} | Test ARI {:.4f}'.format(*test_metrics))


def evaluate(model, gs, features, labels, mask, score):
    model.eval()
    with torch.no_grad():
        logits = model(gs, features)
    return score(logits[mask], labels[mask])


def main():
    parser = argparse.ArgumentParser('HAN Node Classification or Clustering')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', choices=['acm', 'dblp', 'imdb'], default='acm', help='dataset')
    parser.add_argument(
        '--hetero', action='store_true', help='Use heterogeneous graph dataset', default=True
    )
    parser.add_argument('--num-hidden', type=int, default=8, help='number of hidden units')
    parser.add_argument(
        '--num-heads', type=int, default=8, help='number of attention heads in node-level attention'
    )
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout probability')
    parser.add_argument('--task', choices=['clf', 'cluster'], default='clf', help='training task')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='weight decay')
    # parser.add_argument('--device', type=int, default=2, help='GPU 设备')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
