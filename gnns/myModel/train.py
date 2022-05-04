import torch
import argparse
from gnns.utils import set_random_seed, get_device, load_data, add_node_feat
from gnns.myModel.model import MyModel
import torch.optim as optim


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)
    # add_node_feat(g, 'pretrained', args.node_embed_path, True)
    add_node_feat(g, 'random')
    # features = g.nodes[predict_ntype].data['feat']

    model = MyModel(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes}, 
        args.num_hidden, g.canoncial_etypes, args.layer, args.dropout
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * args.epochs, eta_min=args.lr / 100
    )
    pass


def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument('--device', type=int, default=0, help='GPU 设备')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--dataset', choices=['ogbn-mag', 'oag-venue'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='The hidden dim')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--tau', type=float, default=0.8, help='温度参数')
    parser.add_argument('--epochs', type=int, default=150, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=512, help='批大小')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
