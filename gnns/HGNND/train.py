from statistics import mean
import torch
import argparse
from gnns.HGNND.model import HGNND
import torch.optim as optim
from dgl.sampling import sample_neighbors

from gnns.utils import get_device, set_random_seed, micro_macro_f1_score
from gnns.data import ACMDataset, DBLPFourAreaDataset

DATASET = {
    'acm': ACMDataset,
    'dblp': DBLPFourAreaDataset
}

def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data = DATASET[args.dataset]()
    g = data[0]
    predict_ntype = data.predict_ntype
    features = {ntype: g.nodes[ntype].data['feat'].to(device) for ntype in g.ntypes}
    labels = g.nodes[predict_ntype].data['label']
    train_mask = g.nodes[predict_ntype].data['train_mask']
    val_mask = g.nodes[predict_ntype].data['val_mask']
    test_mask = g.nodes[predict_ntype].data['test_mask']
    num_classes = data.num_classes

    g = g.to(device)

    model = HGNND(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes}, 
        args.num_hidden, args.num_hidden, num_classes, predict_ntype, args.kmax
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    score = micro_macro_f1_score
    metrics = 'Epoch {:d} | Train Loss {:.4f} | Train Micro-F1 {:.4f} '\
        '| Train Macro-F1 {:.4f} | Val Micro-F1 {:.4f} | Val Macro-F1 {:.4f}'
    for epoch in range(args.epochs):
        model.train()
        running_loss = []
        pos_mask = torch.topk(torch.rand(g.nodes(predict_ntype).size()), k=args.pos_num)[1]
        pos_idx = g.nodes(predict_ntype)[pos_mask].cpu().numpy()
        pos_graph = sample_neighbors(g, {predict_ntype: pos_idx}, args.pos_num, edge_dir='out')
        pos_src, pos_dst = [], []
        for e in pos_graph.etypes:
            src, dst = pos_graph.edges(etype=e)
            pos_src.extend(src)
            pos_dst.extend(dst)
        pos_src = torch.stack(pos_src)
        pos_dst = torch.stack(pos_dst)
        neg_src = pos_src.repeat(args.neg_num)
        neg_dst = torch.randint(0, g.number_of_nodes(), neg_src.shape, dtype=torch.long)
        loss, logits = model(g, features, pos_src, pos_dst, neg_src, neg_dst, args.alpha * (epoch+1))
        running_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss = mean(running_loss)
        train_metrics = score(logits[train_mask], labels[train_mask])
        val_metrics = evaluate(model, g, features, labels, val_mask, score)
        print(metrics.format(epoch, mean_loss, *train_metrics, *val_metrics))
    test_metrics = evaluate(model, g, features, labels, test_mask, score)
    print('Test Micro-F1 {:.4f} | Test Macro-F1 {:.4f}'.format(*test_metrics))


def evaluate(model, g, features, labels, mask, score):
    model.eval()
    with torch.no_grad():
        _, logits = model(g, features)
    return score(logits[mask], labels[mask])


def main():
    parser = argparse.ArgumentParser(description="训练")
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['dblp', 'acm'], default='acm', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--epochs', type=int, default=1200, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--lr', type=float, default=0.0008, help='学习率')
    parser.add_argument('--kmax', type=float, default=0.001, help='loss dropout率')
    parser.add_argument('--pos-num', type=int, default=3, help='正例个数')
    parser.add_argument('--neg-num', type=int, default=3, help='负例个数')
    parser.add_argument('--alpha', type=float, default=0.00001, help='Loss dropout超参')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()