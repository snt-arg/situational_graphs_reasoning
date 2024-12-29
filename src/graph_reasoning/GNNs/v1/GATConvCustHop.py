from torch_geometric.nn import GATConv
import torch
import torch.nn.init as init
import torch.nn.functional as F


class GATConvCustHop(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, nodes_hidden_channels, edges_hidden_channels, heads, dropout):
        super().__init__()
        ### Nodes
        self.nodes_GATConv1 = GATConv(in_channels_nodes, nodes_hidden_channels, heads=heads, dropout=dropout)
        self.att_l = torch.nn.Parameter(torch.Tensor(1, heads, nodes_hidden_channels))
        
        ### Edges
        self.edges_lin1 = torch.nn.Linear(2 * nodes_hidden_channels + in_channels_edges, edges_hidden_channels)
        self.init_lin_weights()

    def init_lin_weights(self):
        if isinstance(self.edges_lin1, torch.nn.Linear):
            init.xavier_uniform_(self.edges_lin1.weight)
            if self.edges_lin1.bias is not None:
                init.zeros_(self.edges_lin1.bias)
    
    def forward(self, x_dict, edge_index, edge_attr):
        # x = F.dropout(x, p=0.6, training=self.training)
        x1 = F.elu(self.nodes_GATConv1(x_dict, edge_index, edge_attr= edge_attr))
        x1_mean = x1.view(x1.size(0), self.nodes_GATConv1.heads, -1).mean(dim=1)  # [num_nodes, nodes_hidden_channels]

        # Aggregate node embeddings for each edge
        source_nodes_attr = x1_mean[edge_index[0]]  # Embeddings of source nodes
        target_nodes_attr = x1_mean[edge_index[1]]  # Embeddings of target nodes

        # Concatenate source, target node embeddings with edge attributes
        edge_features = torch.cat([source_nodes_attr, target_nodes_attr, edge_attr], dim=1)
        # print("Source nodes shape:", source_nodes_attr)
        # print("Target nodes shape:", target_nodes_attr.shape)
        # print("Edge attributes shape:", edge_attr.shape)
        # print("Concatenated edge features shape:", edge_features.shape)
        # print("Expected input dimension for edges_lin1:", self.edges_lin1.weight.shape[1])
        
        # Update edge embeddings using the linear layer
        edge_attr1 = F.elu(self.edges_lin1(edge_features))
        
        return x1, edge_attr1