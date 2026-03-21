import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, n_classes, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, n_classes)
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
    def __init__(self, nfeat, nhid, n_classes, dropout):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(nfeat, nhid)
        self.layer2 = GCNLayer(nhid, n_classes)
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
        self.a1 = nn.Parameter(torch.empty(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.empty(size=(out_features, 1)))
        
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: [N, in_features]
        adj: Sparse COO Tensor (must have .indices())
        """
        Wh = self.W(h) # [N, out_features]
        N = Wh.size(0)

        # Extract edge indices (row -> source, col -> target)
        indices = adj.indices() # [2, E]
        row, col = indices[0], indices[1]

        # Compute scores per node
        f1 = torch.matmul(Wh, self.a1).squeeze() # [N]
        f2 = torch.matmul(Wh, self.a2).squeeze() # [N]

        # Compute edge scores e_ij for existing edges only
        # f1[row] gives the score for the source of each edge
        # f2[col] gives the score for the target of each edge
        edge_e = self.leakyrelu(f1[row] + f2[col]) # [E]
        
        # Sparse Softmax (over neighbors)
        # We need to compute exp(e_ij) and divide by the sum of exp for each target node
        edge_e_exp = torch.exp(edge_e - torch.max(edge_e)) # [E]
        
        # Summing exponentials per target node (row)
        # We use index_add to accumulate exp values into a buffer of size N
        exp_sum = torch.zeros((N,), device=h.device)
        exp_sum.index_add_(0, row, edge_e_exp) # Accumulate sum for each row i
        
        # Calculate attention coefficients alpha_ij
        # Adding a tiny epsilon to avoid division by zero
        alpha = edge_e_exp / (exp_sum[row] + 1e-10) # [E]
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # Aggregation (Summing alpha_ij * Wh_j)
        # We multiply Wh by alpha for each edge, then accumulate back to source nodes
        weighted_Wh = Wh[col] * alpha.view(-1, 1) # [E, out_features]
        
        h_prime = torch.zeros((N, self.out_features), device=h.device)
        h_prime.index_add_(0, row, weighted_Wh) # [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, is_pubmed=False):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.is_pubmed = is_pubmed

        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)
        ])

        if self.is_pubmed:
            # Paper uses 8 heads averaged for output on Pubmed
            self.out_attentions = nn.ModuleList([
                GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False) 
                for _ in range(nheads)
            ])
        else:
            self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Multi-head concat
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        if self.is_pubmed:
            out = torch.stack([att(x, adj) for att in self.out_attentions], dim=0)
            x = torch.mean(out, dim=0)
        else:
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