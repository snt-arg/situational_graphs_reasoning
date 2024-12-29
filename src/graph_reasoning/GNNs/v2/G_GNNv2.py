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
        hidden_channels = settings["gnn"]["encoder"]["nodes"]["hidden_channels"]
        heads = settings["gnn"]["encoder"]["nodes"]["heads"]
        dropout = settings["gnn"]["encoder"]["nodes"]["dropout"]
        aggr = settings["gnn"]["encoder"]["aggr"]
        training_edge_type = [tuple((e[0],"training",e[2])) for e in settings["hdata"]["edges"]][0]
        metadata = (settings["hdata"]["nodes"], [training_edge_type])
        self.encoder = GNNEncoder(in_channels_nodes, in_channels_edges, hidden_channels, heads[0], dropout, aggr)
        # self.encoder_1 = GATConvCustHop(in_channels_nodes,in_channels_edges,nodes_hidden_channels, edges_hidden_channels, heads[0], dropout)
        # self.encoder_1 = to_hetero(self.encoder_1, metadata, aggr=aggr)

        # ### GNN 2
        # in_channels_edges = edges_hidden_channels + 2 * nodes_hidden_channels * heads[0]
        # # out_nodes_hc = settings["gnn"]["encoder"]["nodes"]["hidden_channels"][-1]
        # # out_edges_hc = settings["gnn"]["encoder"]["edges"]["hidden_channels"][-1]
        # self.encoder_2 = GATConvCustHop(nodes_hidden_channels*heads[0], in_channels_edges,nodes_hidden_channels, edges_hidden_channels, heads[1], dropout)
        # # metadata = (settings["hdata"]["nodes"], [training_edge_type])
        # self.encoder_2 = to_hetero(self.encoder_2, metadata, aggr=aggr)

        ### Decoder
        in_channels_decoder = hidden_channels*2 + hidden_channels*2
        self.decoder = EdgeDecoderMulticlass(settings["gnn"]["decoder"], in_channels_decoder)

    
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
        # print(f"dbg z_node {z_nodes.shape}")
        # print(f"dbg z_edge {z_edges.shape}")
        # print(f"dbg edge_index {edge_index.shape}")
        x = self.decoder(z_nodes, z_edges, edge_index_dict, edge_label_index_tuples_compressed)

        return x