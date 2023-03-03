import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from gnns.myModel.model_metapaths import MyModel
from gnns.utils import (METRICS_STR, add_node_feat, calc_metrics, get_device,
                        load_data, set_random_seed, hg_metapaths, add_label_node)


def train(args):
    if args.seed is not None:
        set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device, reverse_self=True)
    add_node_feat(g, 'pretrained', args.node_embed_path, True)
    # add_node_feat(g, 'random')
    g = add_label_node(g, predict_ntype, labels[train_idx], train_idx)

    all_mps = hg_metapaths(g.to(torch.device('cpu')), predict_ntype, args.num_layers+1)
    layer_metapaths = {
        l: [] for l in range(args.num_layers)
    }
    for mp in all_mps:
        if len(mp.split('-')) - 2 < 0: continue
        layer_metapaths[len(mp.split('-'))-2].append(mp)

    mpnums = len([mp for mp in all_mps if mp.split('-')[0] == predict_ntype]) - 1 # remove 'paper-label'

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
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    optimizer = optim.AdamW(model.parameters(), eps=1e-6)
    stage_labels = labels.clone()
    for s_idx, stage in enumerate(args.stages):
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, args.lr, epochs=stage, steps_per_epoch=len(train_loader),
            pct_start=0.05, anneal_strategy='linear', final_div_factor=10.0
        )
        for epoch in range(stage):
            model.train()
            losses = []
            for input_nodes, output_nodes, blocks in tqdm(train_loader):
                train_labels = stage_labels[output_nodes[predict_ntype]]
                train_logits = model(blocks, blocks[0].srcdata['feat'])
                loss = F.cross_entropy(train_logits, train_labels)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
            if epoch % args.eval_every == 0 or epoch == stage - 1:
                raw_pred, info = evaluate(
                    model, loader, g, labels, data.num_classes, predict_ntype,
                    train_idx, val_idx, test_idx, evaluator
                )
                print(METRICS_STR.format(*info))
        # 选择高置信度节点加入训练集
        if args.save_path:
            torch.save(model.state_dict(), args.save_path + '_stage_' + str(s_idx))
        if s_idx == len(args.stages) - 1:
            break
        pred_prob = raw_pred.softmax(dim=1)
        pred = raw_pred.argmax(dim=-1)
        confidence_mask = pred_prob.max(1)[0] > args.threshold
        enhance_idx = torch.cat([val_idx[confidence_mask[val_idx]], test_idx[confidence_mask[test_idx]]])
        stage_labels[enhance_idx] = pred[enhance_idx]
        # g.add_edges(enhance_idx, pred[enhance_idx], etype='has_rev')
        train_loader = NodeDataLoader(g, {predict_ntype: torch.cat([train_idx, enhance_idx])}, sampler, device=device, batch_size=args.batch_size)

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
        # torch.cuda.empty_cache()
    return logits, calc_metrics(logits, labels, train_idx, val_idx, test_idx, evaluator)


def main():
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument('--device', type=int, default=1, help='GPU 设备')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')
    parser.add_argument('--dataset', choices=['ogbn-mag', 'oag-venue', 'oag-field'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='The hidden dim')
    parser.add_argument('--num-layers', type=int, default=3, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=1, help='训练epoch数')
    parser.add_argument('--stages', nargs='+', type=int, default=[100, 50], help='每个stage的epoch数')
    parser.add_argument('--threshold', type=float, default=0.7, help='多阶段训练置信度阈值')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数量')
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
