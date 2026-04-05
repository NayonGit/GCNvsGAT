import argparse
import time
import torch
import torch.optim as optim
import scipy.sparse as sp
import numpy as np
from models import MLP, GCN, GAT, PyG_GCN, PyG_GAT
from utils import load_data, accuracy, normalize_adjacency

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

def run_single_train(args, data_bundle, device):
    """One training run for a given model and dataset. Returns the test accuracy at the best validation epoch."""
    features, labels, train_mask, val_mask, test_mask, adj_tensor, edge_index = data_bundle
    nfeat = features.shape[1]
    n_classes = int(labels.max().item()) + 1
    
    if args.model == 'MLP':
        model = MLP(nfeat, args.hidden, n_classes, args.dropout)
    elif args.model == 'GCN':
        model = GCN(nfeat, args.hidden, n_classes, args.dropout)
    elif args.model == 'GAT':
        is_pubmed = (args.dataset == 'pubmed')
        model = GAT(nfeat, args.hidden, n_classes, args.dropout, alpha=0.2, nheads=args.heads, is_pubmed=is_pubmed)
    elif args.model == 'PYG_GCN':
        model = PyG_GCN(nfeat, args.hidden, n_classes, args.dropout)
    elif args.model == 'PYG_GAT':
        model = PyG_GAT(nfeat, args.hidden, n_classes, args.dropout, alpha=0.2, nheads=args.heads)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.NLLLoss()

    best_val_acc = 0
    test_acc_at_best_val = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        if args.model == 'MLP':
            output = model(features)
        elif args.model.startswith('PYG_'):
            output = model(features, edge_index)
        else:
            output = model(features, adj_tensor)
            
        loss_train = criterion(output[train_mask], labels[train_mask])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if args.model == 'MLP':
                out = model(features)
            elif args.model.startswith('PYG_'):
                out = model(features, edge_index)
            else:
                out = model(features, adj_tensor)
            
            acc_val = accuracy(out[val_mask], labels[val_mask])
            
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                test_acc_at_best_val = accuracy(out[test_mask], labels[test_mask]).item()

    return test_acc_at_best_val

def main():
    parser = argparse.ArgumentParser(description='GNN Benchmark - 50 Runs Statistics')
    parser.add_argument('--model', type=str, default='GCN', choices=['MLP', 'GCN', 'GAT','PYG_GCN','PYG_GAT'])
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer','pubmed'])
    parser.add_argument('--runs', type=int, default=50, help='Nombre d\'entraînements à moyenner')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--heads', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_path = "./cora/cora/" if args.dataset == 'cora' else f"./{args.dataset}/"
    if args.dataset == 'pubmed': data_path = "pubmed"
    
    adj_gcn, adj_gat, features, labels, train_mask, val_mask, test_mask, _ = load_data(path=data_path, return_raw_adj=True)
    
    features, labels = features.to(device), labels.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    if 'GCN' in args.model:
        adj_tensor = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_gcn)).to(device)
    else:
        adj_tensor = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_gat)).to(device)
    
    edge_index = adj_tensor.indices() if args.model.startswith('PYG_') else None
    
    data_bundle = (features, labels, train_mask, val_mask, test_mask, adj_tensor, edge_index)

    print(f"--- Starting Benchmark: {args.runs} runs of {args.model} on {args.dataset.upper()} ---")
    
    results = []
    start_bench = time.time()
    
    for i in range(args.runs):
        acc = run_single_train(args, data_bundle, device)
        results.append(acc)
        if (i + 1) % 5 == 0:
            print(f"Run {i+1}/{args.runs} completed. Current Mean: {np.mean(results)*100:.2f}%")

    # Final Stats
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    max_acc = np.max(results) * 100
    
    print("\n" + "="*30)
    print(f"FINAL RESULTS for {args.model} on {args.dataset.upper()}")
    print(f"Mean Accuracy: {mean_acc:.2f}%")
    print(f"Std Deviation: {std_acc:.4f}%")
    print(f"Max Accuracy:  {max_acc:.2f}%")
    print(f"Total Time:    {time.time() - start_bench:.2f}s")
    print("="*30)

if __name__ == "__main__":
    main()