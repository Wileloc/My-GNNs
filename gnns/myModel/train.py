import torch
import argparse
from gnns.utils import set_random_seed, get_device, load_data, add_node_feat, METRICS_STR, evaluate
from gnns.myModel.model import MyModel
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
import warnings
import tqdm
import torch.nn.functional as F


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)
    # add_node_feat(g, 'pretrained', args.node_embed_path, True)
    add_node_feat(g, 'random')

    sampler = MultiLayerNeighborSampler(list(range(args.neighbor_size, args.neighbor_size + args.num_layers)))
    train_loader = NodeDataLoader(g, {predict_ntype: train_idx}, sampler, device=device, batch_size=args.batch_size)
    loader = NodeDataLoader(g, {predict_ntype: g.nodes(predict_ntype)}, sampler, device=device, batch_size=args.batch_size)

    model = MyModel(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes}, 
        args.num_hidden, g.canoncial_etypes, args.layer, args.dropout
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * args.epochs, eta_min=args.lr / 100
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            batch_logits = model(blocks, blocks[0].srcdata['feat'])
            batch_labels = labels[output_nodes[predict_ntype]]
            loss = F.cross_entropy(batch_logits, batch_labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
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
