import argparse
import json
import os
import time
import torch
import torch.optim as optim
import scipy.sparse as sp
import numpy as np

from models import MLP, GCN, GAT
from utils import load_data, accuracy, add_graph_noise, normalize_adjacency

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

def main():
    parser = argparse.ArgumentParser(description='GNN Training Pipeline')
    parser.add_argument('--model', type=str, default='GCN', choices=['MLP', 'GCN', 'GAT'])
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer','pubmed'],
                        help='Dataset to use (cora, citeseer or pubmed)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads (GAT only)')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise rate to add to the graph (0.0 to 1.0)')
    args = parser.parse_args()

    model_dir = "best_models"
    history_dir = "history"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running {args.model} on {args.dataset.upper()} ---")
    print(f"--- Running on device: {device} ---")

    is_pubmed = False
    if args.dataset == 'cora':
        data_path = "./cora/cora/"
    elif args.dataset == 'citeseer':
        data_path = "./citeseer/"
    else: 
        data_path = "pubmed"    # Load data
        is_pubmed = True

    # use the load_data func to get all the required stuff
    adj_gcn_raw, adj_gat_raw, features, labels, train_mask, val_mask, test_mask, raw_adj = load_data(path=data_path, return_raw_adj=True)

    # Data to device
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    if args.noise > 0:
        print(f"Training with {args.noise*100}% of added noise")
        raw_adj = add_graph_noise(raw_adj, noise_level=args.noise)
        
        if args.model == 'GCN':
            adj_gcn_raw = normalize_adjacency(raw_adj)
        elif args.model == 'GAT':
            adj_gat_raw = (raw_adj + sp.eye(raw_adj.shape[0]))
 
    # Prepare Adjacency Tensor based on model
    if args.model == 'GCN':
        # Using your sparse conversion function
        adj_tensor = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_gcn_raw)).to(device)
    elif args.model == 'GAT':
        adj_tensor = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_gat_raw)).to(device)
    else:
        adj_tensor = None

    # Model Initialization
    nfeat = int(features.shape[1])
    n_classes = int(labels.max().item()) + 1
    
    if args.model == 'MLP':
        model = MLP(nfeat, args.hidden, n_classes, args.dropout)
    elif args.model == 'GCN':
        model = GCN(nfeat, args.hidden, n_classes, args.dropout)
    elif args.model == 'GAT':
        # Keeping your GAT parameters
        model = GAT(nfeat, args.hidden, n_classes, args.dropout, alpha=0.2, nheads=args.heads, is_pubmed = is_pubmed)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.NLLLoss()

    # Training Loop (Using your logic)
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    noise_suffix = f"_noise_{args.noise}" if args.noise > 0 else ""
    checkpoint_path = os.path.join(model_dir, f'best_{args.model.lower()}_{args.dataset}{noise_suffix}.pth')
    history_filename = os.path.join(history_dir, f'history_{args.model.lower()}_{args.dataset}{noise_suffix}.json')

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Integrated your is_mlp logic style here
        if args.model == 'MLP':
            output = model(features)
        else:
            output = model(features, adj_tensor)
            
        loss_train = criterion(output[train_mask], labels[train_mask])
        acc_train = accuracy(output[train_mask], labels[train_mask])
        
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if args.model == 'MLP':
                output = model(features)
            else:
                output = model(features, adj_tensor)
            loss_val = criterion(output[val_mask], labels[val_mask])
            acc_val = accuracy(output[val_mask], labels[val_mask])
        
        history['train_loss'].append(loss_train.item())
        history['train_acc'].append(acc_train.item())
        history['val_loss'].append(loss_val.item())
        history['val_acc'].append(acc_val.item())

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss Train: {loss_train.item():.4f}, '
                  f'Acc Train: {acc_train:.4f}, Acc Val: {acc_val:.4f}')

    print(f"Training finished in {time.time() - start_time:.2f}s. Best Val Acc: {best_val_acc:.4f}")

    # Save history
    with open(history_filename, 'w') as f:
        json.dump(history, f)

    # Final Test
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        output = model(features) if args.model == 'MLP' else model(features, adj_tensor)
        acc_test = accuracy(output[test_mask], labels[test_mask])
        print(f"FINAL TEST SET ACCURACY: {acc_test:.4f}")

if __name__ == "__main__":
    main()