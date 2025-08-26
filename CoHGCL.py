import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from   scipy.sparse import coo_matrix
from   sklearn.model_selection import KFold
from   sklearn.metrics import roc_auc_score, average_precision_score
from   tqdm import tqdm

import torch
import torch.nn.functional as F

from Model        import CoHGCLModel
from DataHandler  import DataHandler
from Params       import args


SAMPLE_NUM = 1168


def predict(model, label_matrix, trna_count, disease_count):
    pos_preds, neg_preds = [], []
    shuffled_ids = np.random.permutation(trna_count)
    total_samples = len(shuffled_ids)
    steps = int(np.ceil(total_samples / args.batch))

    model.eval()
    with torch.no_grad():
        for step in range(steps):
            start, end = step * args.batch, min((step + 1) * args.batch, total_samples)
            batch_ids  = shuffled_ids[start:end]

            label_matrix_csr = label_matrix.tocsr()
            labels = label_matrix_csr[batch_ids]

            batch_size    = len(batch_ids)
            sample_length = batch_size * 2 * SAMPLE_NUM
            user_indices  = [None] * sample_length
            item_indices  = [None] * sample_length
            current       = 0
            half          = sample_length // 2

            for i in range(batch_size):
                positive_set = np.where(labels[i] != 0)[0]
                sample_num   = min(SAMPLE_NUM, len(positive_set))
                if sample_num == 0:
                    pos = np.random.choice(disease_count, 1)
                    neg = pos
                else:
                    pos = np.random.choice(positive_set, sample_num)
                    neg = []
                    while len(neg) < sample_num:
                        cand = np.random.choice(disease_count)
                        if labels[i][cand] == 0:
                            neg.append(cand)

                for j in range(sample_num):
                    user_indices[current]           = user_indices[current + half] = batch_ids[i]
                    item_indices[current]           = pos[j]
                    item_indices[current + half]    = neg[j]
                    current += 1

            users = torch.tensor(user_indices[:current] + user_indices[half:half+current],
                                 dtype=torch.long, device=args.device)
            items = torch.tensor(item_indices[:current] + item_indices[half:half+current],
                                 dtype=torch.long, device=args.device)

            preds, *_ = model(users, items)

            sample_num_per_batch = users.size(0) // 2
            pos_preds.extend(preds[:sample_num_per_batch].cpu().numpy())
            neg_preds.extend(preds[sample_num_per_batch:sample_num_per_batch].cpu().numpy())

    labels = [1] * len(pos_preds) + [0] * len(neg_preds)
    all_preds = np.concatenate([pos_preds, neg_preds])
    return labels, all_preds


def main():

    os.makedirs(args.results_dir, exist_ok=True)

    pos_data = pd.read_csv(args.positive_csv, header=None).values
    neg_data = pd.read_csv(args.negative_csv, header=None).values

    trna_count    = args.trna_count
    disease_count = args.disease_count
    kf            = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    fold_aucs, fold_auprs = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(pos_data), start=1):
        print(f"\n=== Fold {fold}/{args.n_splits} ===")

        train_pos = pos_data[train_idx]
        test_pos  = pos_data[test_idx]

        val_size  = int(len(train_pos) * 0.1)
        val_pos   = train_pos[:val_size]
        train_pos = train_pos[val_size:]

        train_mat = np.vstack((train_pos, neg_data))
        val_mat   = val_pos
        test_neg_idx = np.random.choice(len(neg_data), size=len(test_pos), replace=False)
        test_mat = np.vstack((test_pos, neg_data[test_neg_idx]))

        train_matrix = coo_matrix(
            (np.ones(len(train_mat)), (train_mat[:, 0], train_mat[:, 1])),
            shape=(trna_count, disease_count)).astype(np.float32)
        val_matrix   = coo_matrix(
            (np.ones(len(val_mat)), (val_mat[:, 0], val_mat[:, 1])),
            shape=(trna_count, disease_count)).astype(np.float32)
        test_matrix  = coo_matrix(
            (np.ones(len(test_mat)), (test_mat[:, 0], test_mat[:, 1])),
            shape=(trna_count, disease_count)).astype(np.float32)

        data_handler = DataHandler(train_matrix, val_matrix, test_matrix)
        data_handler.load_data()

        def to_tensor(sparse_mat):
            m  = sp.coo_matrix(sparse_mat)
            idx = torch.from_numpy(np.vstack((m.row, m.col)).astype(np.int64))
            dat = torch.from_numpy(m.data) if m.data.size > 0 else torch.tensor([0.0])
            if idx.shape[1] == 0:
                idx = torch.tensor([[0, 0]])
                dat = torch.tensor([0.0])
            return torch.sparse_coo_tensor(idx, dat, m.shape)

        adj_tensor          = to_tensor(data_handler.adj)
        transpose_adj_tensor = to_tensor(data_handler.transpose_adj)

        model     = CoHGCLModel(adj_tensor, transpose_adj_tensor,
                                data_handler.tRNA_count, data_handler.disease_count).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()

        for ep in tqdm(range(1, args.epoch + 1), desc=f"Fold {fold} Training", leave=False):
            model.train()
            shuffled_ids = np.random.permutation(trna_count)
            epoch_loss   = 0.0

            steps = int(np.ceil(len(shuffled_ids) / args.batch))
            for step in range(steps):
                s, e = step * args.batch, min((step + 1) * args.batch, len(shuffled_ids))
                batch_ids = shuffled_ids[s:e]

                label_csr = data_handler.get_train_matrix().tocsr()[batch_ids].toarray()

                batch_size    = len(batch_ids)
                sample_length = batch_size * 2 * SAMPLE_NUM
                u_idx = [None] * sample_length
                i_idx = [None] * sample_length
                cur   = 0
                half  = sample_length // 2

                for i in range(batch_size):
                    pos_set   = np.where(label_csr[i] != 0)[0]
                    samp_num  = min(SAMPLE_NUM, len(pos_set))
                    if samp_num == 0:
                        pos = np.random.choice(disease_count, 1)
                        neg = pos
                    else:
                        pos = np.random.choice(pos_set, samp_num)
                        neg = []
                        while len(neg) < samp_num:
                            cand = np.random.choice(disease_count)
                            if label_csr[i][cand] == 0:
                                neg.append(cand)

                    for j in range(samp_num):
                        u_idx[cur]           = u_idx[cur + half] = batch_ids[i]
                        i_idx[cur]           = pos[j]
                        i_idx[cur + half]    = neg[j]
                        cur += 1

                users = torch.tensor(u_idx[:cur] + u_idx[half:half+cur], dtype=torch.long, device=args.device)
                items = torch.tensor(i_idx[:cur] + i_idx[half:half+cur], dtype=torch.long, device=args.device)

                preds, _, _, _, _, contrast = model(users, items)

                num_pairs = users.size(0) // 2
                targets   = torch.cat([torch.ones(num_pairs, device=args.device),
                                       torch.zeros(num_pairs, device=args.device)])

                bce_loss  = bce_loss_fn(preds, targets)
                loss      = bce_loss + args.contrast_weight * contrast

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            val_labels, val_preds = predict(model,
                                            data_handler.get_val_matrix(),
                                            trna_count, disease_count)
            val_auc  = roc_auc_score(val_labels, val_preds)
            val_aupr = average_precision_score(val_labels, val_preds)
            print(f"Ep {ep:02d}  Loss {epoch_loss/steps:.4f}  "
                  f"Val AUC {val_auc:.4f}  Val AUPR {val_aupr:.4f}")

        test_labels, test_preds = predict(model,
                                          data_handler.get_test_matrix(),
                                          trna_count, disease_count)
        test_auc  = roc_auc_score(test_labels, test_preds)
        test_aupr = average_precision_score(test_labels, test_preds)
        fold_aucs.append(test_auc)
        fold_auprs.append(test_aupr)

        print(f"[Fold {fold}] Test AUC: {test_auc:.4f}  Test AUPR: {test_aupr:.4f}")

    mean_auc,  std_auc  = np.mean(fold_aucs),  np.std(fold_aucs)
    mean_aupr, std_aupr = np.mean(fold_auprs), np.std(fold_auprs)

    print("\n===== 5-Fold CV Result =====")
    for i, (a, p) in enumerate(zip(fold_aucs, fold_auprs), start=1):
        print(f"Fold {i}:  AUC {a:.4f}  AUPR {p:.4f}")
    print(f"Mean AUC  : {mean_auc:.4f}  ± {std_auc:.4f}")
    print(f"Mean AUPR : {mean_aupr:.4f}  ± {std_aupr:.4f}")


if __name__ == "__main__":
    main()
