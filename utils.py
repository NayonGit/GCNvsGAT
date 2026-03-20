import numpy as np
import scipy.sparse as sp
import torch
import os
import pandas as pd

#function for noise experiment
def add_graph_noise(adj, noise_level=0.1):
    """
    Add random edges. 
    noise_level 0.1 = we add 10% of edges.
    """
    adj_coo = adj.tocoo()
    num_nodes = adj.shape[0]
    # Total number of edges to add
    num_noise = int(adj_coo.nnz * noise_level)
    
    # Random idx 
    noise_row = np.random.randint(0, num_nodes, num_noise)
    noise_col = np.random.randint(0, num_nodes, num_noise)
    noise_data = np.ones(num_noise)
    
    # Noise adjacency matrix
    noise_adj = sp.coo_matrix((noise_data, (noise_row, noise_col)), 
                              shape=(num_nodes, num_nodes), dtype=np.float32)
    
    # Combination with the original one
    combined_adj = adj + noise_adj
    combined_adj = combined_adj + combined_adj.T
    combined_adj.data = np.ones_like(combined_adj.data) 
    
    return combined_adj

def remove_graph_edges(adj, drop_rate=0.2):
    """
    Randomly removes a percentage of existing edges.
    """
    adj_coo = adj.tocoo()
    mask = np.random.rand(adj_coo.nnz) > drop_rate
    
    new_row = adj_coo.row[mask]
    new_col = adj_coo.col[mask]
    new_data = adj_coo.data[mask]
    
    new_adj = sp.coo_matrix((new_data, (new_row, new_col)), 
                            shape=adj.shape, dtype=np.float32)
    # Add self-loops back for GAT stability
    return new_adj + sp.eye(adj.shape[0])

def normalize_adjacency(adj):
    """
    Returns the normalized adjacency matrix A_hat = D^-1/2 * (A + I) * D^-1/2
    Exactly as per your notebook.
    """
    adj = sp.coo_matrix(adj)
    adj_tilde = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_tilde.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj_tilde).dot(d_mat_inv_sqrt).tocoo()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_data(path="./cora/cora/", return_raw_adj=False):
    """
    Loads data from Cora or CiteSeer datasets.
    Automatically detects the dataset type and normalizes features.
    """
    path = path.lower()
    # Identify dataset name from path to locate files
    if "pubmed" in path:
        dataset_name = "pubmed"
    elif "cora" in path:
        dataset_name = "cora"
    elif "citeseer" in path:
        dataset_name = "citeseer"
    else:
        raise ValueError(f"Dataset not existing in : {path}. "
                         "The path must contain: 'cora', 'citeseer' or 'pubmed'.")
    if dataset_name == "pubmed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='.', name="Pubmed")
        data = dataset[0]
        
        features = data.x
        labels = data.y
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        
        # Adjacency Matrix
        edge_index = data.edge_index.numpy()
        adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                            shape=(data.num_nodes, data.num_nodes), dtype=np.float32)
    else:

        content_file = os.path.join(path, f"{dataset_name}.content")
        cites_file = os.path.join(path, f"{dataset_name}.cites")

        # Load .content file
        # CiteSeer has 3703 features, Cora has 1433. We load without fixed names.
        node_data = pd.read_csv(content_file, sep='\t', header=None)

        # Extract and Normalize Features
        # Features are between ID (col 0) and Label (last col)
        raw_features = node_data.iloc[:, 1:-1].values
        
        # Row normalization (sum to 1) - Crucial for GAT stability
        row_sums = raw_features.sum(1)
        row_sums[row_sums == 0] = 1.0
        raw_features = raw_features / row_sums[:, np.newaxis]
        
        features = torch.FloatTensor(raw_features)

        # Labels
        labels_raw = node_data.iloc[:, -1].values
        unique_labels = np.unique(labels_raw)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        labels = torch.LongTensor([label_map[l] for l in labels_raw])

        # Build Adjacency Matrix (Robust to String/Int IDs)
        idx = node_data.iloc[:, 0].values
        idx_map = {j: i for i, j in enumerate(idx)}
        
        edges_unordered = pd.read_csv(cites_file, sep='\t', header=None).values
        
        # Filter edges where IDs exist in the content file
        edges = []
        for edge in edges_unordered:
            u, v = idx_map.get(edge[0]), idx_map.get(edge[1])
            if u is not None and v is not None:
                edges.append([u, v])
        edges = np.array(edges)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # Masks (More robust version)
        N = labels.shape[0]
        indices = np.arange(N)
        
        # We fix the random seed for mask generation to be reproducible
        np.random.seed(42) 
        np.random.shuffle(indices)

        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)

        # Standard Planetoid split proportions for CiteSeer
        train_idx = indices[:120] # CiteSeer standard is often 120 nodes
        val_idx = indices[120:620]
        test_idx = indices[N-1000:]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
    
    # Symmetrize: A = A + A.T
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    # Normalizations
    adj_norm = normalize_adjacency(adj)
    adj_gat = (adj + sp.eye(adj.shape[0])).todense()


    if return_raw_adj:
        return adj_norm, adj_gat, features, labels, train_mask, val_mask, test_mask, adj
    return adj_norm, adj_gat, features, labels, train_mask, val_mask, test_mask