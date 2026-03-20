import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.projection = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, adj_norm):
        # Feature transform
        x = self.projection(x)
        
        # Propagation
        if adj_norm.is_sparse:
            x = torch.spmm(adj_norm, x)
        else:
            x = torch.mm(adj_norm, x)
        return x

class GCN(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, dropout):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(n_features, n_hidden)
        self.layer2 = GCNLayer(n_hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        x = F.relu(self.layer1(x, adj_norm))
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.layer2(x, adj_norm)
        return F.log_softmax(x, dim=1)

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector split in two: for source and target
        self.a1 = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.empty(size=(out_features, 1)))
        
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h) # [N, out_features]
        
        # Compute attention scores e
        # f1: score for nodes as 'source', f2: score for nodes as 'target'
        f1 = torch.matmul(Wh, self.a1) # [N, 1]
        f2 = torch.matmul(Wh, self.a2) # [N, 1]
        
        # Broadcast sum to get all-pairs combinations: e_ij = LeakyReLU(a1*Wh_i + a2*Wh_j)
        e = self.leakyrelu(f1 + f2.T) # [N, N]

        # Masking (Exactly like your notebook)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Normalization
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Aggregation
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)
        ])

        # Output layer
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # Multi-head concat
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        # Output log_softmax
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)

# this function is used to modify freely the architecture of the GAT model
def load_gat_model_autodetect(checkpoint_path, nfeat=1433, nclass=7, device='cpu'):
    """
    Automatically detects GAT architecture from the checkpoint file.
    """
    # Load the state_dict first to inspect it
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Extract architecture from keys
    # Count how many 'attentions.X' keys exist to find nheads
    head_keys = [k for k in state_dict.keys() if 'attentions.' in k and '.W.weight' in k]
    nheads = len(head_keys)
    
    # Get the shape of the first weight matrix to find nhid
    # Shape of W.weight is [nhid, nfeat]
    first_head_weight = state_dict[head_keys[0]]
    nhid = first_head_weight.shape[0]
    
    print(f"DEBUG: Detected GAT with nheads={nheads}, nhid={nhid}")
    
    # Initialize the model with detected parameters
    model = GAT(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.6, alpha=0.2, nheads=nheads)
    
    # Load the weights
    model.load_state_dict(state_dict)
    return model.to(device)