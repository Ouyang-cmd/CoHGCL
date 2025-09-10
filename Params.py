import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Model Parameters')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--latdim', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--n_splits', type=int)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--positive_csv', type=str)
    parser.add_argument('--negative_csv', type=str)
    parser.add_argument('--trna_count', type=int)
    parser.add_argument('--disease_count', type=int)
    parser.add_argument('--contrast_weight', type=float)
    parser.add_argument('--rwr_restart', type=float)
    parser.add_argument('--rwr_iter', type=int)
    parser.add_argument('--gat_dropout', type=float)
    parser.add_argument('--gat_alpha', type=float)
    parser.add_argument('--hyper_num', type=int)

    return parser.parse_args()

args = parse_args()
args.device = f"cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
