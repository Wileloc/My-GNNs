import argparse
import random
import warnings

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from gnns.data import ACMDataset
from gnns.hgt.model import HGT
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, f1_score,
                             normalized_mutual_info_score)
from utils import set_random_seed

DATASET = {
    'acm': ACMDataset
}


def train(args):
    set_random_seed(args.seed)
    data = DATASET[args.dataset]()
    g = data[0]
    predict_ntype = data.predict_ntype
    features = {ntype: g.nodes[ntype].data['feat'] for ntype in g.ntypes}
    labels = g.nodes[predict_ntype].data['label']
    train_mask = g.nodes[predict_ntype].data['train_mask']
    val_mask = g.nodes[predict_ntype].data['val_mask']
    test_mask = g.nodes[predict_ntype].data['test_mask']

    model = HGT(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_heads, g.ntypes, g.canonical_etypes,
        predict_ntype, args.num_layers, args.dropout
    )
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, total_steps=args.epochs)
    metrics = 'Epoch {:d} | Train Loss {:.4f} | Train Micro-F1 {:.4f} | Train Macro-F1 {:.4f}' \
              ' | Val Micro-F1 {:.4f} | Val Macro-F1 {:.4f}' \
              ' | Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        logits, embeds = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        train_scores = micro_macro_f1_score(logits[train_mask], labels[train_mask])
        val_scores = evaluate(model, g, features, labels, val_mask, micro_macro_f1_score)
        test_scores = evaluate(model, g, features, labels, test_mask, micro_macro_f1_score)
        print(metrics.format(epoch, loss.item(), *train_scores, *val_scores, *test_scores))
    test_scores = evaluate(model, g, features, labels, test_mask, micro_macro_f1_score)
    print('Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(*test_scores))
    nmi, ari = node_cluster(embeds[predict_ntype].numpy(), labels.numpy(), data.num_classes())
    print('Test NMI {:.4f} | Test ARI {:.4f}'.format(nmi, ari))


@torch.no_grad()
def evaluate(model, g, features, labels, mask, score):
    model.eval()
    logits = model(g, features)
    return score(logits[mask], labels[mask])


def micro_macro_f1_score(logits, labels):
    """计算Micro-F1和Macro-F1得分

    :param logits: tensor(N, C) 预测概率，N为样本数，C为类别数
    :param labels: tensor(N) 正确标签
    :return: float, float Micro-F1和Macro-F1得分
    """
    prediction = torch.argmax(logits, dim=1).long().numpy()
    labels = labels.numpy()
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return micro_f1, macro_f1

@torch.no_grad()
def node_cluster(embeds, labels, num_classes):
    seeds = [random.randint(0, 0x7fffffff) for _ in range(10)]
    nmi, ari = [], []
    for seed in seeds:
        pred = KMeans(num_classes, random_state=seed).fit_predict(embeds)
        nmi.append(normalized_mutual_info_score(labels, pred))
        ari.append(adjusted_rand_score(labels, pred))
    return sum(nmi) / len(nmi), sum(ari) / len(ari)


def main():
    parser = argparse.ArgumentParser(description='HGT Node Classification')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', choices=['acm', 'imdb'], default='acm', help='dataset')
    parser.add_argument('--num-hidden', type=int, default=256, help='number of hidden units')
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--max-lr', type=float, default=1e-3, help='upper learning rate boundary')
    parser.add_argument('--clip', type=float, default=1.0, help='gradient norm clipping')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
