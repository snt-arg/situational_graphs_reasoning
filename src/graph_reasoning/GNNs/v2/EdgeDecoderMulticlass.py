import torch
import torch.nn.init as init
import copy


class EdgeDecoderMulticlass(torch.nn.Module):
    def __init__(self, settings, in_channels, dropout):
        super().__init__()
        hidden_channels = settings["hidden_channels"]
        self.decoder_lins = torch.nn.ModuleList()
        self.decoder_lins.append(torch.nn.Linear(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            self.decoder_lins.append(torch.nn.Linear(hidden_channels[i], hidden_channels[i+1]))
        self.decoder_lins.append(torch.nn.Linear(hidden_channels[-1], settings["output_channels"]))
        for decoder_lin in self.decoder_lins:
            self.init_lin_weights(decoder_lin)

    def init_lin_weights(self,model):
        if isinstance(model, torch.nn.Linear):
            init.xavier_uniform_(model.weight)
            if model.bias is not None:
                init.zeros_(model.bias)
    

    def forward(self, z_nodes, z_edges, edge_index_dict, edge_label_dict):
        ### Data gathering
        node_key = list(edge_index_dict.keys())[0][0]
        edge_key = list(edge_index_dict.keys())[0][1]
        edge_index = copy.copy(edge_index_dict[node_key, edge_key, node_key]).cpu().numpy()
        # edge_index_tuples = list(zip(edge_index[0], edge_index[1]))

        # edge_label_index = copy.copy(edge_label_index_dict[node_key, edge_key, node_key]).cpu().numpy()
        # edge_label_index_tuples_compressed = np.array(list({tuple(sorted((edge_label_index[0, i], edge_label_index[1, i]))) for i in range(edge_label_index.shape[1])}))
        # edge_label_index_tuples_compressed_inversed = edge_label_index_tuples_compressed[:, ::-1]
        # src, dst = edge_label_index_tuples_compressed[:,0], edge_label_index_tuples_compressed[:,1]

        # edge_index_to_edge_label_index = [np.argwhere((edge_label_index_single == edge_index_tuples).all(1))[0][0] for edge_label_index_single in edge_label_index_tuples_compressed]
        # edge_index_to_edge_label_index_inversed = [np.argwhere((edge_label_index_single == edge_index_tuples).all(1))[0][0] for edge_label_index_single in edge_label_index_tuples_compressed_inversed]
        # ### Network forward
        z = torch.cat([z_nodes[edge_label_dict["src"]], z_nodes[edge_label_dict["dst"]], z_edges[edge_label_dict["edge_index_to_edge_label_index"]], z_edges[edge_label_dict["edge_index_to_edge_label_index_inversed"]]], dim=-1) ### ONLY NODE AND EDGE EMBEDDINGS
        for decoder_lin in self.decoder_lins[:-1]:
            z = decoder_lin(z).relu()
        z = self.decoder_lins[-1](z)

        return z