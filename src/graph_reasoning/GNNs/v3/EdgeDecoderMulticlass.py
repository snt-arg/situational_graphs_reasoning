import torch
import torch.nn.init as init
import torch.nn.functional as F
import copy



class EdgeDecoderMulticlass(torch.nn.Module):
    def __init__(self, settings, in_channels, dropout):
        super().__init__()
        hidden_channels = settings["common"]
        self.dropout = dropout
        self.settings = settings

        self.decoder_common_lins = torch.nn.ModuleList()
        self.decoder_common_lins.append(torch.nn.Linear(in_channels, settings["common"]["hidden_channels"][0]))
        for i in range(len(settings["common"]["hidden_channels"]) - 1):
            self.decoder_common_lins.append(torch.nn.Linear(settings["common"]["hidden_channels"][i], settings["common"]["hidden_channels"][i+1]))
        self.decoder_common_lins.append(torch.nn.Linear(settings["common"]["hidden_channels"][-1], settings["common"]["output_channels"]))
        for decoder_lin in self.decoder_common_lins:
            self.init_lin_weights(decoder_lin)

        self.decoder_classifier_lins = torch.nn.ModuleList()
        self.decoder_classifier_lins.append(torch.nn.Linear(settings["common"]["output_channels"], settings["classifier"]["hidden_channels"][0]))
        for i in range(len(settings["classifier"]["hidden_channels"]) - 1):
            self.decoder_classifier_lins.append(torch.nn.Linear(settings["classifier"]["hidden_channels"][i], settings["classifier"]["hidden_channels"][i+1]))
        self.decoder_classifier_lins.append(torch.nn.Linear(settings["classifier"]["hidden_channels"][-1], settings["classifier"]["classes"]))
        for decoder_lin in self.decoder_classifier_lins:
            self.init_lin_weights(decoder_lin)

        self.decoder_uncertainty_lins = torch.nn.ModuleList()
        self.decoder_uncertainty_lins.append(torch.nn.Linear(settings["common"]["output_channels"], settings["uncertainty"]["hidden_channels"][0]))
        for i in range(len(settings["uncertainty"]["hidden_channels"]) - 1):
            self.decoder_uncertainty_lins.append(torch.nn.Linear(settings["uncertainty"]["hidden_channels"][i], settings["uncertainty"]["hidden_channels"][i+1]))
        self.decoder_uncertainty_lins.append(torch.nn.Linear(settings["uncertainty"]["hidden_channels"][-1], settings["uncertainty"]["uncertainty_logits"]))
        for decoder_lin in self.decoder_uncertainty_lins:
            self.init_lin_weights(decoder_lin)

    def init_lin_weights(self,model):
        if isinstance(model, torch.nn.Linear):
            init.xavier_uniform_(model.weight)
            if model.bias is not None:
                init.zeros_(model.bias)
    

    def forward(self, z_nodes, z_edges, edge_index_dict, edge_label_dict):
        self.use_dropout = True if self.training or self.use_MC_dropout else False

        z = torch.cat([z_nodes[edge_label_dict["src"]], z_nodes[edge_label_dict["dst"]], z_edges[edge_label_dict["edge_index_to_edge_label_index"]], z_edges[edge_label_dict["edge_index_to_edge_label_index_inversed"]]], dim=-1) ### ONLY NODE AND EDGE EMBEDDINGS
        for decoder_lin in self.decoder_common_lins:
            z = decoder_lin(z).relu()
            z = F.dropout(z, p=self.dropout, training=self.use_dropout)

        zc = z
        zu = z
        
        for decoder_lin in self.decoder_classifier_lins[:-1]:
            zc = decoder_lin(zc).relu()
            zc = F.dropout(zc, p=self.dropout, training=self.use_dropout)
        zc = self.decoder_classifier_lins[-1](zc)


        for decoder_lin in self.decoder_uncertainty_lins[:-1]:
            zu = decoder_lin(zu).relu()
            zu = F.dropout(zu, p=self.dropout, training=self.use_dropout)
        zu = self.decoder_uncertainty_lins[-1](zu) # This is s = log(sigma^2)
        return zc, zu
