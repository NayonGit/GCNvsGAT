import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json
import numpy as np
import scipy.sparse as sp
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from models import MLP, GCN, GAT
from utils import load_data
from train import sparse_mx_to_torch_sparse_tensor


def plot_metrics_from_json(file_path):
    """
    Loads the history from a JSON file and displays Accuracy and Loss curves.
    """
    filename = os.path.basename(file_path)
    parts = filename.replace('.json', '').split('_')
    model_name = parts[1].upper() if len(parts) > 1 else "Unknown"
    dataset = parts[2].upper() if len(parts) > 1 else "Unknown"
    try:
        with open(file_path, 'r') as f:
            h = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Did you download it from the server?")
        return

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax[0].plot(h['train_acc'], label='Train Acc', color='blue')
    ax[0].plot(h['val_acc'], label='Val Acc', color='orange')
    ax[0].set_title(f'Accuracy Evolution - {model_name.upper()} - {dataset}')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.6)
    
    # Loss
    ax[1].plot(h['train_loss'], label='Train Loss', color='blue')
    ax[1].plot(h['val_loss'], label='Val Loss', color='orange')
    ax[1].set_title(f'Loss Evolution - {model_name.upper()} - {dataset}')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def show_confusion_matrix(model, features, labels, mask, model_name, adj=None, dataset = "Cora"):
    model.eval()
    
    if adj is not None and sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    with torch.no_grad():
        if model_name.upper() == 'MLP':
            output = model(features)
        else:
            if adj is None:
                raise ValueError(f"Adjacency matrix 'adj' is required for {model_name}")
            output = model(features, adj)
        
        preds = output.max(1)[1]
    
    y_true = labels[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()
    
    print(f"\n--- Classification Report: {model_name.upper()} ---")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name.upper()} - {dataset}')
    plt.show()

def visualize_tsne(model, features, labels, model_name, adj=None, dataset = "Cora"):
    model.eval()
    
    if adj is not None :
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        
    with torch.no_grad():
        if model_name.upper() == 'MLP':
            x = model.fc1(features)
            embeddings = F.relu(x)
        elif model_name.upper() == 'GCN':
            embeddings = F.relu(model.layer1(features, adj))
        elif model_name.upper() == 'GAT':
            x = torch.cat([att(features, adj) for att in model.attentions], dim=1)
            embeddings = F.elu(x)
        else:
            raise ValueError("Unknown model type")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    out = tsne.fit_transform(embeddings.cpu().numpy())
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(out[:, 0], out[:, 1], c=labels.cpu().numpy(), 
                          cmap='rainbow', s=30, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f"t-SNE Embeddings - {model_name.upper()} - {dataset}")
    plt.show()