from torch_geometric.loader import HGTLoader
from torch_geometric.datasets import OGB_MAG
import torch_geometric.transforms as T
import torch
from tqdm import tqdm
from gnns.utils import get_device, set_random_seed

import argparse


def train(args):
    if args.seed is not None:
        set_random_seed(args.seed)
    device = get_device(args.device)

    transform = T.ToUndirected(merge=True)
    dataset = OGB_MAG('data/pyg_data/MAG', preprocess='metapath2vec', transform=transform)
    train_loader = HGTLoader(
        dataset[0],
        num_samples={key: [1800] * args.num_layers for key in dataset[0].node_types},
        batch_size=args.batch_size,
        input_nodes=('paper', dataset[0]['paper'].train_mask)
    )
    # loader = HGTLoader(
    #     dataset[0],
    #     num_samples={key: [1800] * args.num_layers for key in dataset[0].node_types},
    #     batch_size=args.batch_size,
    #     input_nodes=('paper')
    # )
    
    for batch in tqdm(train_loader):
        batch = batch.to(device, 'edge_index')
        print(batch)


def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument('--device', type=int, default=1, help='GPU 设备')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--dataset', choices=['ogbn-mag', 'oag-venue', 'oag-field'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='The hidden dim')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=40, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=2048, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=5, help='邻居采样数量')
    parser.add_argument('--drop-rate', type=float, default=0.5, help='丢失概率')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--add-feat-method', choices=['random', 'pretrained'], default='random', help='顶点特征添加方式')
    parser.add_argument('--eval-every', type=int, default=10, help='每多少个epoch计算一次准确率')
    parser.add_argument('--save-path', help='模型保存路径')
    parser.add_argument('--node-embed-path', default='model/word2vec/ogbn-mag.model', help='预训练顶点嵌入路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
