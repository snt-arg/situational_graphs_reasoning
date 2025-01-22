import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops


class EGATConv(MessagePassing):
    def __init__(self, in_channels_nodes, in_channels_edges, out_channels, heads=1, dropout=0.0, aggr="add"):
        super().__init__(aggr=aggr)  # Aggregation method: "add"

        self.in_channels_nodes = in_channels_nodes
        self.in_channels_edges = in_channels_edges
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        # print(f"dbg in_channels_nodes {in_channels_nodes}")
        # print(f"dbg in_channels_edges {in_channels_edges}")
        # print(f"dbg out_channels {out_channels}")
        # print(f"dbg heads {heads}")
        # print(f"dbg dropout {dropout}")

        # Node transformation
        self.node_fc = torch.nn.Linear(in_channels_nodes, heads * out_channels, bias=False)
        # Edge transformation
        self.edge_fc = torch.nn.Linear(in_channels_edges, heads * out_channels, bias=False)
        # Attention mechanism
        self.att_fc = torch.nn.Linear(2 * out_channels + out_channels, 1, bias=False)  # For node-pair and edge concat        

        self.edge_node_fc = torch.nn.Linear(2 * out_channels, out_channels, bias=False)
        self.edge_update_fc = torch.nn.Linear(3 * out_channels, out_channels, bias=False)
        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.node_fc.weight)
        glorot(self.edge_fc.weight)
        glorot(self.att_fc)
        if hasattr(self, 'bias') and self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0)
        x = self.node_fc(x)  # Shape: [num_nodes, heads * out_channels]
        edge_attr = self.edge_fc(edge_attr)  # Shape: [num_edges, heads * out_channels]

        # Propagate messages
        node_embeddings = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Update edge attributes
        node_embeddings_i, node_embeddings_j = node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]  # Source and target node embeddings
        edge_attr = self.update_edge_features(node_embeddings_i, node_embeddings_j, edge_attr)

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        return node_embeddings, edge_attr

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # # Reshape for multi-head attention
        x_i = x_i.view(-1, self.heads, self.out_channels)  # Shape: [num_nodes, heads, out_channels]
        x_j = x_j.view(-1, self.heads, self.out_channels)  # Shape: [num_nodes, heads, out_channels]
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)  # Shape: [num_edges, heads, out_channels]

        # Compute attention scores
        att_input = torch.cat([x_i, x_j, edge_attr], dim=-1)  # Concatenate along feature dimension
        # print(f"att_input shape: {att_input.shape}")
        alpha = self.att_fc(att_input).squeeze(-1)  # Compute attention scores [num_edges, heads]
        # print(f"alpha 1 shape: {alpha.shape}")
        alpha = F.leaky_relu(alpha)
        # print(f"alpha 2 shape: {alpha.shape}")
        alpha = softmax(alpha, index, ptr, size_i)  # Apply softmax over neighbors
        # print(f"alpha 3 shape: {alpha.shape}")
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # print(f"alpha 4 shape: {alpha.shape}")

        # Combine x_j and edge_attr in the message
        combined = torch.cat([x_j, edge_attr], dim=-1)  # [num_edges, heads, 2 * out_channels]
        weighted_message = self.edge_node_fc(combined) * alpha.unsqueeze(-1)  # [num_edges, heads, out_channels]
        # print(f"weighted_message shape before aggregation: {weighted_message.shape}")

        # Aggregate across heads
        weighted_message = weighted_message.mean(dim=1)  # [num_edges, out_channels]
        # print(f"weighted_message after head aggregation: {weighted_message.shape}")

        # Return weighted messages
        return weighted_message

    def update(self, aggr_out):
        # Return aggregated messages
        return aggr_out
    
    def update_edge_features(self, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)  # Shape: [num_edges, heads, out_channels]
        edge_attr = edge_attr.mean(dim=1)
        # Concatenate source, target node embeddings and edge attributes
        combined = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [num_edges, heads, 3 * out_channels]
        new_edge_attr = self.edge_update_fc(combined)  # [num_edges, heads, out_channels]

        # Optional: Residual connection
        # new_edge_attr = new_edge_attr + edge_attr

        # Apply activation or normalization
        new_edge_attr = F.relu(new_edge_attr)  # [num_edges, heads, out_channels]

        return new_edge_attr

# Integrate into GNNEncoder
class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, hidden_channels, heads, dropout, aggr = "add"):
        super().__init__()
        self.heads = heads
        self.egat1 = EGATConv(
            in_channels_nodes=in_channels_nodes,
            in_channels_edges=in_channels_edges,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            aggr = aggr
        )
        self.egat2 = EGATConv(
            in_channels_nodes=hidden_channels,
            in_channels_edges=in_channels_edges,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            aggr = aggr
        )

    def forward(self, x, edge_index, edge_attr):

        # Update node embeddings using EGAT layers
        x1, edge_attr1 = self.egat1(x, edge_index, edge_attr)
        x1 = F.elu(x1)
        edge_attr1 = F.relu(edge_attr1)
        x1, edge_attr1 = self.egat2(x1, edge_index, edge_attr)
        x1 = F.elu(x1)
        edge_attr1 = F.relu(edge_attr1)

        return x1, edge_attr1
