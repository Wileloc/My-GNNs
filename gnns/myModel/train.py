import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from gnns.myModel.model import MyModel
from gnns.utils import (METRICS_STR, add_node_feat, evaluate, get_device,
                        load_data, set_random_seed)


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)
    print("开始添加顶点特征")
    add_node_feat(g, 'pretrained', args.node_embed_path, True)
    # add_node_feat(g, 'random')

    sampler = MultiLayerNeighborSampler(list(range(args.neighbor_size, args.neighbor_size + args.num_layers)))
    train_loader = NodeDataLoader(g, {predict_ntype: train_idx}, sampler, device=device, batch_size=args.batch_size)
    loader = NodeDataLoader(g, {predict_ntype: g.nodes(predict_ntype)}, sampler, device=device, batch_size=args.batch_size)

    model = MyModel(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes}, 
        args.num_hidden, data.num_classes, g.canonical_etypes, args.num_layers, args.dropout
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=args.lr / 100
    )
    alpha = 1 - args.alpha
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            # blocks 表示对每一层的采样，逻辑应该是从这一批次要学习的顶点开始，依次采样倒数第一层，倒数第二层...的节点。
            batch_logits, batch_co_loss = model(blocks, blocks[0].srcdata['feat'])
            batch_labels = labels[output_nodes[predict_ntype]]
            sv_loss = F.cross_entropy(batch_logits[predict_ntype], batch_labels)
            loss = alpha * sv_loss + (1 - alpha) * batch_co_loss
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            print(METRICS_STR.format(*evaluate(
                model, loader, g, labels, data.num_classes, predict_ntype,
                train_idx, val_idx, test_idx, evaluator
            )))
    if args.save_path:
        torch.save(model.cpu().state_dict(), args.save_path)
        print('模型已保存到', args.save_path)


def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument('--device', type=int, default=4, help='GPU 设备')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--dataset', choices=['ogbn-mag', 'oag-venue'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='The hidden dim')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--tau', type=float, default=0.8, help='温度参数')
    parser.add_argument('--epochs', type=int, default=40, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=5, help='邻居采样数量')
    parser.add_argument('--alpha', type=float, default=0.1, help='对比损失占比')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--eval-every', type=int, default=10, help='每多少个epoch计算一次准确率')
    parser.add_argument('--save-path', help='模型保存路径')
    parser.add_argument('--node_embed_path', default='model/word2vec/ogbn-mag.model', help='预训练顶点嵌入路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
