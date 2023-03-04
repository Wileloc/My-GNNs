import tqdm
from utils import *
from torch.optim import Adam


def train(model, X, y, A, A_norm, Ad):
    """
    train our model
    Args:
        model: Dual Correlation Reduction Network
        X: input feature matrix
        y: input label
        A: input origin adj
        A_norm: normalized adj
        Ad: graph diffusion
    Returns: acc, nmi, ari, f1
    """
    print("Training…")
    # calculate embedding similarity and cluster centers
    sim, centers = model_init(model, X, y, A_norm)
    centers = torch.tensor(centers).to(opt.args.device)

    # initialize cluster centers
    model.cluster_centers.data = centers

    # edge-masked adjacency matrix (Am): remove edges based on feature-similarity
    # Am = remove_edge(A, sim, remove_rate=0.1)

    # KNN adjacency matrix
    Ak = knn(A, sim)

    T = None
    # hop adj
    # A_norm_d = A_norm.to_dense()
    # A_2_hop = torch.mm(A_norm_d, A_norm_d)
    # A_3_hop = torch.mm(A_2_hop, A_norm_d)
    # T = A_norm_d + A_2_hop + A_3_hop + torch.eye(A_norm_d.shape[0], device=A_norm_d.device)
    # Ad_2_hop = torch.mm(Ad, Ad)
    # Ad_3_hop = torch.mm(Ad_2_hop, Ad)
    # Ad_all = [Ad, Ad_2_hop, Ad_3_hop]
    # Am_2_hop = torch.mm(Am, Am)
    # Am_3_hop = torch.mm(Am_2_hop, Am)
    # Am_all = [Am, Am_2_hop, Am_3_hop]

    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.epoch)):
        # add gaussian noise to X
        X_tilde1, X_tilde2 = gaussian_noised_feature(X)

        # input & output
        X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, Z, AZ_all, Z_all = model(X_tilde1, Ad, X_tilde2, Ak, Ad)

        # clustering & evaluation & affinity matrix
        acc, nmi, ari, f1, z_centers, _ = clustering(Z, y)
        z_centers = torch.from_numpy(z_centers).to(opt.args.device)
        h_c, pseudo_labels = high_confidence(Z, z_centers, threshold=0.04)
        T = h_c + torch.eye(A_norm.shape[0], device=A_norm.device)
        # clustering-level embeddings
        ae1_centers, ae2_centers, igae1_centers, igae2_centers = [], [], [], []
        for i in range(opt.args.n_clusters):
            ae1_centers.append(torch.mean(Z_ae_all[0][pseudo_labels == i], dim=0))
            ae2_centers.append(torch.mean(Z_ae_all[1][pseudo_labels == i], dim=0))
            igae1_centers.append(torch.mean(Z_gae_all[0][pseudo_labels == i], dim=0))
            igae2_centers.append(torch.mean(Z_gae_all[1][pseudo_labels == i], dim=0))
        ae1_centers = torch.stack(ae1_centers, dim=0)
        ae2_centers = torch.stack(ae2_centers, dim=0)
        igae1_centers = torch.stack(igae1_centers, dim=0)
        igae2_centers = torch.stack(igae2_centers, dim=0)
        Z_ae_all.extend([ae1_centers, ae2_centers])
        Z_gae_all.extend([igae1_centers, igae2_centers])

        # calculate loss: L_{DICR}, L_{REC} and L_{KL}
        L_DICR = dicr_loss(Z_ae_all, Z_gae_all, AZ_all, Z_all, T)
        L_REC = reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat)
        L_KL = distribution_loss(Q, target_distribution(Q[0].data))
        loss = L_DICR + L_REC + opt.args.lambda_value * L_KL

        # optimization
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # acc, nmi, ari, f1, _, _ = clustering(Z, y)
        if acc > opt.args.acc:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.f1 = f1

    return opt.args.acc, opt.args.nmi, opt.args.ari, opt.args.f1
