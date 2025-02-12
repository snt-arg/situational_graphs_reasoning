from graph_reasoning.GNNs.v2.EdgeAwareGatConv import EGATConv
import torch

conv = EGATConv(
    in_channels_nodes=64,     # Input node feature size
    in_channels_edges=10,     # Input edge feature size
    out_channels=32,          # Output feature size per head
    heads=4,                  # Number of attention heads
    dropout=0.5               # Dropout rate
)
# Example inputs
x = torch.rand(10, 64)  # 10 nodes, 64 features each
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 3 edges
edge_attr = torch.rand(3, 10)  # 3 edges, 10 features each

# Forward pass
output = conv(x, edge_index, edge_attr)
print("Output shape:", output.shape)  # [num_nodes, heads, out_channels]
