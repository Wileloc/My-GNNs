import opt
import torch
from torch.optim import Adam
import torch.nn.functional as F
# from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans
from utils import adjust_learning_rate
from utils import eva, target_distribution

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = ['usps', 'hhar', 'reut', 'acm', 'dblp', 'cite']


def Train(epoch, model, data, adj, label, lr, pre_model_save_path, final_model_save_path, n_clusters,
          original_acc, gamma_value, lambda_value, device):
    optimizer = Adam(model.parameters(), lr=lr)
    model.load_state_dict(torch.load(pre_model_save_path, map_location=device))
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde = model(data, adj)
    # kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    # cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    cluster_id, cluster_centers = kmeans(z_tilde, num_clusters=n_clusters, device=device)
    model.cluster_layer.data = torch.tensor(cluster_centers).to(device)
    eva(label, cluster_id.detach().cpu().numpy(), 'Initialization')
    final_emb = None
    final_label = None

    for epoch in range(epoch):
        # if opt.args.name in use_adjust_lr:
        #     adjust_learning_rate(optimizer, epoch)
        x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde = model(data, adj)

        tmp_q = q.data
        p = target_distribution(tmp_q)

        loss_ae = F.mse_loss(x_hat, data)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss_igae = loss_w + gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = loss_ae + loss_igae + lambda_value * loss_kl
        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())
        cluster_id, cluster_centers = kmeans(z_tilde, num_clusters=opt.args.n_clusters, device=device)

        acc, nmi, ari, f1 = eva(label, cluster_id.detach().cpu().numpy(), epoch)
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)

        if acc > original_acc:
            original_acc = acc
            # torch.save(model.state_dict(), final_model_save_path)
            final_emb = z_tilde
            final_label = cluster_id
    torch.save(final_emb, f'DFCN_{opt.args.name}_emb.pt')
    torch.save(final_label, f'DFCN_{opt.args.name}_label.pt')
