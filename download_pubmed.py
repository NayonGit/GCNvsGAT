import os
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid

def main():
    print("--- Starting PubMed Download and Processing ---")
    
    # Download PubMed using Torch Geometric
    dataset = Planetoid(root='.', name="Pubmed")
    data = dataset[0]

    # Extract Data
    num_nodes = data.num_nodes
    features = data.x
    labels = data.y
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # Build Adjacency Matrix (COO Format)
    edge_index = data.edge_index.numpy()
    adj_raw = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes), 
        dtype=np.float32
    )

    # Symmetrize 
    adj_raw = adj_raw + adj_raw.T.multiply(adj_raw.T > adj_raw) - adj_raw.multiply(adj_raw.T > adj_raw)

    # Summary
    print("-" * 30)
    print(f"Dataset: PubMed")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {adj_raw.nnz}")
    print(f"Features: {features.shape[1]}")
    print(f"Classes: {dataset.num_classes}")
    print("-" * 30)

    # Saving for further use
    # This avoids re-downloading/re-processing every time
    output_dir = "pubmed_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save({
        'features': features,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }, os.path.join(output_dir, 'data_tensors.pt'))
    
    sp.save_npz(os.path.join(output_dir, 'adj_raw.npz'), adj_raw)
    
    print(f"Files saved in: {output_dir}/")
    print("Done.")

if __name__ == "__main__":
    main()