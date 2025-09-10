import torch
import torch.nn as nn
import torch.nn.functional as F
from Params import args


class BipartiteGraphAttentionLayer(nn.Module):
    def __init__(self, in_src, in_tgt, out, dropout, alpha, concat=True):
        super().__init__()
        self.W_src = nn.Linear(in_src, out,  bias=False)
        self.W_tgt = nn.Linear(in_tgt, out,  bias=False)
        self.a     = nn.Parameter(torch.empty(2 * out, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.leaky  = nn.LeakyReLU(alpha)
        self.dropout = dropout
        self.concat  = concat
        self.out = out

    def forward(self, src, tgt, adj):
        h_i = self.W_src(src)
        h_j = self.W_tgt(tgt)

        src_id, tgt_id = adj._indices()

        e = self.leaky(torch.matmul(torch.cat([h_i[src_id], h_j[tgt_id]], dim=1), self.a)).squeeze()
        e = e.clamp(min=-30., max=30.)
        exp_e = torch.exp(e)

        denom = torch.zeros(src.size(0), device=src.device)
        denom.scatter_add_(0, src_id, exp_e)
        alpha = exp_e / (denom[src_id] + 1e-8)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros_like(h_i)
        out.scatter_add_(0,
                         src_id.unsqueeze(-1).expand(-1, self.out),
                         h_j[tgt_id] * alpha.unsqueeze(-1))
        return F.elu(out) if self.concat else out


def compute_rwr(P, r, k):
    I = torch.eye(P.size(0), device=P.device)
    term, F_mat = I.clone(), I.clone()
    for _ in range(k):
        term  = r * torch.matmul(P, term)
        F_mat = F_mat + term
    return (1 - r) * F_mat


def info_nce_loss(z1, z2, T):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    sim    = torch.matmul(z1, z2.T) / T
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim, labels)


class CoHGCLModel(nn.Module):
    def __init__(self, adj, tadj, trna_n, disease_n):
        super().__init__()
        H, D = args.hyper_num, args.latdim
        init = nn.init.xavier_normal_

        self.trna_embed    = nn.Parameter(init(torch.empty(trna_n,  D, device=args.device)))
        self.disease_embed = nn.Parameter(init(torch.empty(disease_n, D, device=args.device)))

        self.adj, self.tadj = adj, tadj

        self.gat_t_layers = nn.ModuleList([
            BipartiteGraphAttentionLayer(D, D, D, args.gat_dropout, args.gat_alpha)
            for _ in range(3)])
        self.gat_d_layers = nn.ModuleList([
            BipartiteGraphAttentionLayer(D, D, D, args.gat_dropout, args.gat_alpha)
            for _ in range(3)])

        P_t = row_norm(torch.sparse.mm(adj,  tadj).to_dense())
        P_d = row_norm(torch.sparse.mm(tadj, adj ).to_dense())
        self.rwr_t = compute_rwr(P_t, args.rwr_restart, args.rwr_iter)
        self.rwr_d = compute_rwr(P_d, args.rwr_restart, args.rwr_iter)

        self.mlp = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )
        self.h = nn.Parameter(torch.randn(2, device=args.device))

        self.gate_layer = nn.Sequential(
            nn.Linear(args.latdim, 1),
            nn.Sigmoid()
        )

    def forward(self, trna_ids, disease_ids):
        sg_t, sg_d = self.trna_embed, self.disease_embed
        sg_t = self.gat_t_layers[0](sg_t, sg_d, self.adj)
        sg_d = self.gat_d_layers[0](sg_d, sg_t, self.tadj)

        S_t = torch.matmul(sg_t, self.thyper)
        S_d = torch.matmul(sg_d, self.dhyper)
        hg_t = self.hg_t_layers[0](sg_t, S_t)
        hg_d = self.hg_d_layers[0](sg_d, S_d)
        S_t = torch.matmul(hg_t, self.thyper)
        S_d = torch.matmul(hg_d, self.dhyper)
        hg_t = self.hg_t_layers[1](hg_t, S_t)
        hg_d = self.hg_d_layers[1](hg_d, S_d)
        sg_t = self.gat_t_layers[1](hg_t, hg_d, self.adj)
        sg_d = self.gat_d_layers[1](hg_d, hg_t, self.tadj)
        sg_t = self.gat_t_layers[2](sg_t, sg_d, self.adj)
        sg_d = self.gat_d_layers[2](sg_d, sg_t, self.tadj)
        S_t = torch.matmul(hg_t, self.thyper)
        S_d = torch.matmul(hg_d, self.dhyper)
        hg_t = self.hg_t_layers[2](hg_t, S_t)
        hg_d = self.hg_d_layers[2](hg_d, S_d)

        final_t = sg_t + hg_t
        final_d = sg_d + hg_d

        sel_t = final_t.index_select(0, trna_ids)
        sel_d = final_d.index_select(0, disease_ids)
        contrast = info_nce_loss(sg_t, hg_t) + info_nce_loss(sg_d, hg_d)

        gmf_vector = sel_t * sel_d
        gmf_pred = torch.sum(gmf_vector, dim=-1)
        mlp_input = torch.cat([sel_t, sel_d], dim=-1)
        mlp_pred = self.mlp(mlp_input).squeeze(-1)

        gate = self.gate_layer(gmf_vector).squeeze(-1)
        fusion_pred = gate * gmf_pred + (1 - gate) * mlp_pred

        return fusion_pred, sg_t, sg_d, gmf_pred, mlp_pred, contrast


