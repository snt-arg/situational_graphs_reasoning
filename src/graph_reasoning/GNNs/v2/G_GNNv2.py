import torch
from torch_geometric.nn import to_hetero

from graph_reasoning.GNNs.v2.EdgeAwareGatConv import GNNEncoder
from graph_reasoning.GNNs.v2.EdgeDecoderMulticlass import EdgeDecoderMulticlass

class G_GNNv2(torch.nn.Module):
    def __init__(self, settings, logger):
        super().__init__()
        self.logger = logger

        ### GNN 1
        in_channels_nodes = settings["gnn"]["encoder"]["nodes"]["input_channels"]
        in_channels_edges = settings["gnn"]["encoder"]["edges"]["input_channels"]
        nodes_hc = settings["gnn"]["encoder"]["nodes"]["hidden_channels"][0]
        edges_hc = settings["gnn"]["encoder"]["edges"]["hidden_channels"][0]
        heads = settings["gnn"]["encoder"]["nodes"]["heads"]
        dropout = settings["gnn"]["dropout"]
        aggr = settings["gnn"]["encoder"]["aggr"]
        self.encoder = GNNEncoder(in_channels_nodes, in_channels_edges, nodes_hc, edges_hc, heads[0], dropout, aggr)

        ### Decoder
        in_channels_decoder = nodes_hc*2 + edges_hc*2
        self.decoder = EdgeDecoderMulticlass(settings["gnn"]["decoder"], in_channels_decoder, dropout)
    
    
    def set_use_MC_dropout(self,value):
        self.use_MC_dropout = value
        self.encoder.set_use_MC_dropout(value)
        self.decoder.use_MC_dropout = value

    
    def forward(self, x_dict, edge_index_dict, edge_label_index_tuples_compressed):
        node_key = list(edge_index_dict.keys())[0][0]
        edge_key = list(edge_index_dict.keys())[0][1]
        src, dst = edge_index_dict[node_key, edge_key, node_key]
        # z_emb_dict_wn = {(node_key, edge_key, node_key) : torch.cat([x_dict[node_key][src], x_dict[node_key][dst], x_dict[node_key, edge_key, node_key]], dim=1)}
        edge_index_dict[list(edge_index_dict.keys())[0]] = edge_index_dict[list(edge_index_dict.keys())[0]].long()

        x = x_dict[node_key]
        edge_index = edge_index_dict[node_key, edge_key, node_key]
        edge_attr = x_dict[node_key, edge_key, node_key]

        z_nodes, z_edges = self.encoder(x, edge_index, edge_attr)
        x = self.decoder(z_nodes, z_edges, edge_index_dict, edge_label_index_tuples_compressed)
        fake_uncertainty = torch.ones(x.shape[0]).to("cuda:0")
        return x, fake_uncertainty