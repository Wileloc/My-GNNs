import argparse
import warnings
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from gnns.myModel.model_mix import MyModel
from gnns.utils import (METRICS_STR, add_node_feat, calc_metrics, get_device,
                        load_data, set_random_seed, hg_metapaths, accuracy)


def train(args):
    torch.autograd.set_detect_anomaly(True)
    if args.seed is not None:
        set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device, reverse_self=True)
    add_node_feat(g, 'pretrained', args.node_embed_path, True)
    # add_node_feat(g, 'random')

    all_mps = hg_metapaths(g.to(torch.device('cpu')), predict_ntype, args.num_layers+1)
    layer_metapaths = {
        l: [] for l in range(args.num_layers)
    }
    for mp in all_mps:
        if len(mp.split('-')) - 2 < 0: continue
        layer_metapaths[len(mp.split('-'))-2].append(mp)

    mpnums = len([mp for mp in all_mps if mp.split('-')[0] == predict_ntype])

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_layers)
    train_loader = NodeDataLoader(g, {predict_ntype: train_idx}, sampler, device=device, batch_size=args.batch_size)
    loader = NodeDataLoader(g, {predict_ntype: g.nodes(predict_ntype)}, sampler, device=device, batch_size=args.batch_size)

    model = MyModel(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes}, 
        args.num_hidden, data.num_classes, g.canonical_etypes, args.num_layers, predict_ntype, layer_metapaths, mpnums, attn_drop=args.dropout
    ).to(args.device)
    if args.load_path:
        model.load_state_dict(torch.load(args.load_path, map_location=device))
    # if args.load:
    #     model.load_state_dict(torch.load('model_new/myModel_mix_n3_distinct_mp.pt'))
    #     evaluate(model, loader, g, labels, data.num_classes, predict_ntype, train_idx, val_idx, test_idx, evaluator)
    #     exit(0)
    optimizer = optim.AdamW(model.parameters(), eps=1e-6)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05, anneal_strategy='linear', final_div_factor=10.0
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            train_labels = labels[output_nodes[predict_ntype]]
            # train_logits, augment_labels = model(blocks, blocks[0].srcdata['feat'], train_labels)
            # train_labels = torch.cat((train_labels, augment_labels))
            train_logits = model(blocks, blocks[0].srcdata['feat'])
            loss = F.cross_entropy(train_logits, train_labels)
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
            torch.save(model.state_dict(), args.save_path)

    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print('模型已保存到', args.save_path)


@torch.no_grad()
def evaluate(model, loader, g, labels, num_classes, predict_ntype,
        train_idx, val_idx, test_idx, evaluator=None):
    model.eval()
    logits = torch.zeros(g.num_nodes(predict_ntype), num_classes, device=train_idx.device)
    for input_nodes, output_nodes, blocks in loader:
        logits[output_nodes[predict_ntype]] = model(blocks, blocks[0].srcdata['feat'])
        torch.cuda.empty_cache()
    return calc_metrics(logits, labels, train_idx, val_idx, test_idx, evaluator)


def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument('--device', type=int, default=0, help='GPU 设备')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')
    parser.add_argument('--dataset', choices=['ogbn-mag', 'oag-venue', 'oag-field'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='The hidden dim')
    parser.add_argument('--num-layers', type=int, default=3, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=11, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=2048, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=5, help='邻居采样数量')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--eval-every', type=int, default=10, help='每多少个epoch计算一次准确率')
    parser.add_argument('--save-path', help='模型保存路径')
    parser.add_argument('--node-embed-path', default='model/word2vec/ogbn-mag.model', help='预训练顶点嵌入路径')
    parser.add_argument('--load-path', help='是否加载模型')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
