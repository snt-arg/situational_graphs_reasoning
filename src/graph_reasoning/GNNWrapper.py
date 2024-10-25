import numpy as np
import torch, os, sys
from torch_geometric.nn import to_hetero, GATConv
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
import copy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from itertools import compress
from colorama import Fore
import time
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.init as init

# from .FactorNN import FactorNN

# graph_reasoning_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_reasoning")
# sys.path.append(graph_reasoning_dir)
# from graph_reasoning.from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata
# graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
# sys.path.append(graph_datasets_dir)
# from graph_datasets.graph_visualizer import visualize_nxgraph
# graph_matching_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_matching")
# sys.path.append(graph_matching_dir)
from graph_matching.utils import plane_6_params_to_4_params
# graph_factor_nn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_factor_nn")
# sys.path.append(graph_factor_nn_dir)
from graph_factor_nn.FactorNNBridge import FactorNNBridge
from graph_reasoning.from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata
from graph_reasoning.MetricsSubplot import MetricsSubplot
from graph_datasets.graph_visualizer import visualize_nxgraph


class GNNWrapper():
    def __init__(self, settings, report_path, logger = None) -> None:
        print(f"GNNWrapper: ", Fore.BLUE + "Initializing" + Fore.WHITE)
        self.settings = settings
        self.target_concept = settings["report"]["target_concept"]
        self.report_path = report_path
        self.logger = logger
        self.use_gnn_factors = True
        self.pth_path = os.path.join(self.report_path,'model.pth')
        self.best_pth_path = os.path.join(self.report_path,'model_best.pth')
        if self.settings["gnn"]["use_cuda"]:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        if logger:
            self.logger.info(f"GNNWrapper: torch device => {self.device}")
        else:
            print(f"GNNWrapper: ", Fore.BLUE + f"torch device => {self.device}" + Fore.WHITE)
        metric_values_dict = {"loss" : [], "auc" : [], "acc" : [], "prec" : [], "rec" : [], "f1" : [], "pred_pos_rate" : [], "gt_pos_rate": []}
        self.metric_values = {"train" : copy.deepcopy(metric_values_dict), "val" : copy.deepcopy(metric_values_dict),\
                            "test" : copy.deepcopy(metric_values_dict), "inference" : copy.deepcopy(metric_values_dict),}
        if self.use_gnn_factors:
            self.factor_nn = FactorNNBridge(["room", "wall", "floor"])

        plot_names_map = {"train Metrics": 0, "val Metrics": 1, "train RoomWall inference example": 2, "val RoomWall inference example": 3,\
                          "Inference rooms graph": 4, "Inference walls graph": 5}
        self.metric_subplot = MetricsSubplot(nrows=2, ncols=3, plot_names_map=plot_names_map)
        
    def set_dataset(self, nxdataset):
        print(f"GNNWrapper: ", Fore.BLUE + f"Prepocessing nxdataset" + Fore.WHITE)
        self.nxdataset = nxdataset
        self.hdataset = self.preprocess_nxdataset(nxdataset)

    def preprocess_nxdataset(self, nxdataset):
        # lnl_settings = self.settings["link_neighbor_loader"]
        # edge_types = [tuple(e) for e in self.settings["hdata"]["edges"]]
        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]
        # print(f"dbg preprocess_nxdataset edge_types {edge_types}")
        hdataset = {}
        for tag in nxdataset.keys():
            hdataset_key = []
            # for nx_data in nxdataset[tag]:
            for nx_data in (pbar := tqdm.tqdm(nxdataset[tag], colour="blue")):
                hdata = from_networkxwrapper_2_heterodata(nx_data)
                hdataset_key.append(hdata)
                # transform = T.RandomLinkSplit(
                #     num_val=0.0,
                #     num_test=0.0,
                #     key= "edge_label",
                #     neg_sampling_ratio=0.0,
                #     edge_types=edge_types,
                #     is_undirected = False
                # )
                # hdata, _, _ = transform(hdata)
                # print(f"dbg hdata 2 {hdata}")
                # print(f"dbg hdata 2 {hdata[('ws','training', 'ws')]['edge_index']}")
                # loaders_tmp.append( LinkNeighborLoader(
                #     data=hdata,
                #     num_neighbors=lnl_settings["num_neighbors"],
                #     neg_sampling_ratio=0.0,
                #     neg_sampling = None,
                #     edge_label_index=(tuple(edge_types), hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
                #     edge_label=hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label,
                #     batch_size=lnl_settings["batch_size"],
                #     shuffle=lnl_settings["shuffle"],
                #     subgraph_type="induced"
                # ))
                # print(f"dbg loaders_tmp[0] {loaders_tmp[0]}")
                # print(f"dbg hdata 1 {loaders_tmp[0].data[('ws','training', 'ws')]['edge_label']}")

            hdataset[tag] = hdataset_key

        # ### Plots
        # for tag in self.nxdataset.keys():
        #     last_graph = copy.deepcopy(self.nxdataset[tag][-1])
        #     edge_index = loaders[tag][-1].data[edge_types[0],edge_types[1],edge_types[2]].edge_index.cpu().numpy()
        #     edge_label_index = loaders[tag][-1].data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.cpu().numpy()

        #     ### Message passing
        #     edge_index_tuples = [(edge_index[0][i], edge_index[1][i]) for i in range(len(edge_index[0]))]
        #     mp_edges_last_graph = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "brown", "linewidth":1.0, "alpha":0.8}) for i,pair in enumerate(edge_index_tuples)]
        #     merged_graph = self.merge_predicted_edges(copy.deepcopy(last_graph), mp_edges_last_graph)
        #     # visualize_nxgraph(merged_graph, image_name = f"{tag} inference example - message passing")

        #     if self.settings["report"]["save"]:
        #         plt.savefig(os.path.join(self.report_path,f'{tag}_inference_example-mp.png'), bbox_inches='tight')

        #     ### Ground truth
        #     gt_in_loader = []
        #     input_id_in_samples = []

        #     for sampled_data in loaders[tag][-1]:
        #         input_id_in_samples = input_id_in_samples + list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].input_id.cpu().numpy())
        #         gt_in_loader = gt_in_loader + list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.cpu().numpy())

        #     color_code = ["black", "blue", "red"]
        #     original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        #     predicted_edges_last_graph = [(edge_label_index[0][j], edge_label_index[1][j], {"type" : original_edge_types[gt_in_loader[i]],\
        #                                 "label": gt_in_loader[i], "viz_feat": color_code[gt_in_loader[i]], "linewidth":0.5 if gt_in_loader[i]==0 else 1.5,\
        #                                 "alpha":0.3 if gt_in_loader[i]==0 else 1.}) for i, j in enumerate(input_id_in_samples)]
            
        #     merged_graph = self.merge_predicted_edges(copy.deepcopy(last_graph), predicted_edges_last_graph)
        #     # visualize_nxgraph(merged_graph, image_name = f"{tag} inference example - ground truth")
        #     # plt.show()
        #     # time.sleep(999)
        #     if self.settings["report"]["save"]:
        #         plt.savefig(os.path.join(self.report_path,f'{tag}_inference_example-ground_truth.png'), bbox_inches='tight')

        return hdataset

            
    def define_GCN(self):
        print(f"GNNWrapper: ", Fore.BLUE + "Defining GCN" + Fore.WHITE)
            
        class GNNEncoder(torch.nn.Module):
            def __init__(self, in_channels_nodes, in_channels_edges, nodes_hidden_channels, edges_hidden_channels, heads, dropout):
                super().__init__()
                ### Nodes
                self.nodes_GATConv1 = GATConv(in_channels_nodes, nodes_hidden_channels, heads=heads, dropout=dropout)
                self.att_l = torch.nn.Parameter(torch.Tensor(1, heads, nodes_hidden_channels))
                
                ### Edges
                self.edges_lin1 = torch.nn.Linear(in_channels_edges, edges_hidden_channels)
                self.init_lin_weights()

            def init_lin_weights(self):
                if isinstance(self.edges_lin1, torch.nn.Linear):
                    init.xavier_uniform_(self.edges_lin1.weight)
                    if self.edges_lin1.bias is not None:
                        init.zeros_(self.edges_lin1.bias)
            
            def forward(self, x_dict, edge_index, edge_weight, edge_attr):
                # x = F.dropout(x, p=0.6, training=self.training)
                x1 = F.elu(self.nodes_GATConv1(x_dict, edge_index, edge_attr= edge_attr))
                edge_attr1 = self.edges_lin1(edge_attr)
                
                return x1, edge_attr1


        class EdgeDecoderMulticlass(torch.nn.Module):
            def __init__(self, settings, in_channels):
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
            

            def forward(self, z_dict, z_emb_dict, edge_index_dict, edge_label_index_dict):

                ### Data gathering
                node_key = list(edge_index_dict.keys())[0][0]
                edge_key = list(edge_index_dict.keys())[0][1]
                src, dst = edge_label_index_dict[node_key, edge_key, node_key]
                edge_label_index = edge_label_index_dict[node_key, edge_key, node_key]
                edge_index = copy.copy(edge_index_dict[node_key, edge_key, node_key]).cpu().numpy()
                edge_index_tuples = np.array(list(zip(edge_index[0], edge_index[1])))
                edge_label_index = copy.copy(edge_label_index).cpu().numpy()
                edge_label_index_tuples = np.array(list(zip(edge_label_index[0], edge_label_index[1])))

                edge_index_to_edge_label_index = [np.argwhere((edge_label_index_single == edge_index_tuples).all(1))[0][0] for edge_label_index_single in edge_label_index_tuples]

                ### Network forward
                z = torch.cat([z_dict[node_key][src], z_dict[node_key][dst], z_emb_dict[node_key,edge_key, node_key][edge_index_to_edge_label_index]], dim=-1) ### ONLY NODE AND EDGE EMBEDDINGS
                for decoder_lin in self.decoder_lins[:-1]:
                    z = decoder_lin(z).relu()
                z = self.decoder_lins[-1](z)

                return z


        class Model(torch.nn.Module):
            def __init__(self, settings, logger):
                super().__init__()
                self.logger = logger

                ### GNN 1
                in_channels_nodes = settings["gnn"]["encoder"]["nodes"]["input_channels"]
                in_channels_edges = settings["gnn"]["encoder"]["edges"]["input_channels"] + 2 * in_channels_nodes
                nodes_hidden_channels = settings["gnn"]["encoder"]["nodes"]["hidden_channels"]
                edges_hidden_channels = settings["gnn"]["encoder"]["edges"]["hidden_channels"]
                heads = settings["gnn"]["encoder"]["nodes"]["heads"]
                dropout = settings["gnn"]["encoder"]["nodes"]["dropout"]
                aggr = settings["gnn"]["encoder"]["aggr"]
                self.encoder_1 = GNNEncoder(in_channels_nodes,in_channels_edges,nodes_hidden_channels, edges_hidden_channels, heads[0], dropout)
                training_edge_type = [tuple((e[0],"training",e[2])) for e in settings["hdata"]["edges"]][0]
                metadata = (settings["hdata"]["nodes"], [training_edge_type])
                self.encoder_1 = to_hetero(self.encoder_1, metadata, aggr=aggr)

                ### GNN 2
                in_channels_edges = edges_hidden_channels + 2 * nodes_hidden_channels * heads[0]
                # out_nodes_hc = settings["gnn"]["encoder"]["nodes"]["hidden_channels"][-1]
                # out_edges_hc = settings["gnn"]["encoder"]["edges"]["hidden_channels"][-1]
                self.encoder_2 = GNNEncoder(nodes_hidden_channels*heads[0], in_channels_edges,nodes_hidden_channels, edges_hidden_channels, heads[1], dropout)
                # metadata = (settings["hdata"]["nodes"], [training_edge_type])
                self.encoder_2 = to_hetero(self.encoder_2, metadata, aggr=aggr)

                ### Decoder
                in_channels_decoder = nodes_hidden_channels*2 + edges_hidden_channels
                self.decoder = EdgeDecoderMulticlass(settings["gnn"]["decoder"], in_channels_decoder)

            
            def forward(self, x_dict, edge_index_dict, edge_label_index_dict):
                node_key = list(edge_index_dict.keys())[0][0]
                edge_key = list(edge_index_dict.keys())[0][1]
                src, dst = edge_index_dict[node_key, edge_key, node_key]
                z_emb_dict_wn = {(node_key, edge_key, node_key) : torch.cat([x_dict[node_key][src], x_dict[node_key][dst], x_dict[node_key, edge_key, node_key]], dim=1)}
                edge_index_dict[list(edge_index_dict.keys())[0]] = edge_index_dict[list(edge_index_dict.keys())[0]].long()
                z_dict, z_emb_dict = self.encoder_1(x_dict, edge_index = edge_index_dict, edge_weight = None, edge_attr = z_emb_dict_wn)
                z_emb_dict_wn = {(node_key, edge_key, node_key) : torch.cat([z_dict[node_key][src], z_dict[node_key][dst], z_emb_dict[node_key, edge_key, node_key]], dim=1)}
                z_dict, z_emb_dict = self.encoder_2(z_dict, edge_index = edge_index_dict, edge_weight = None, edge_attr = z_emb_dict_wn)
                x = self.decoder(z_dict, z_emb_dict, edge_index_dict, edge_label_index_dict)
                return x
            
        self.model = Model(self.settings, self.logger)

    def train(self, verbose = False):
        print(f"GNNWrapper: ", Fore.BLUE + "Training" + Fore.WHITE)

        best_val_loss = float('inf')
        patience = 100
        trigger_times = 0

        training_settings = self.settings["training"]
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings["gnn"]["lr"])
        self.criterion = torch.nn.CrossEntropyLoss()
        original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]

        # if verbose:
        #     plt.show(block=False)

        for epoch in (pbar := tqdm.tqdm(range(1, training_settings["epochs"]), colour="blue")):
            pbar.set_description(f"Epoch")
            total_loss = total_examples = 0
            gt_in_train_dataset, probs_in_train_dataset = [], []

            for i, hdata in enumerate(self.hdataset["train"]):
                self.optimizer.zero_grad()
                gt = hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label.to(self.device)
                gt_in_train_dataset = gt_in_train_dataset + list(gt.cpu().numpy())

                hdata.to(self.device)
                logits = self.model(hdata.x_dict, hdata.edge_index_dict,\
                                    hdata.edge_index_dict)

                loss = self.criterion(logits, gt)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss) * logits.numel()
                total_examples += logits.numel()
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                probs_in_train_dataset = probs_in_train_dataset + [list(probs)]
                preds = np.argmax(probs, axis=1)

                if i == len(self.hdataset["train"]) - 1:
                    edge_index = list(hdata[edge_types[0],edge_types[1],edge_types[2]].edge_index.cpu().numpy())
                    edge_index = np.array(list(zip(edge_index[0], edge_index[1])))

                    color_code = ["black", "blue", "red"]
                    predicted_edges_last_graph = [(ei[0], ei[1], {"type" : original_edge_types[preds[i]],\
                                                "label": preds[i], "viz_feat": color_code[preds[i]], "linewidth":0.5 if preds[i]==0 else 1.5,\
                                                "alpha":0.3 if preds[i]==0 else 1.}) for i, ei in enumerate(edge_index)]
            
            probs_in_train_dataset = np.concatenate(probs_in_train_dataset, axis=0)
            accuracy, precission, recall, f1, auc = self.compute_metrics_from_all_predictions(gt_in_train_dataset, probs_in_train_dataset, verbose= False)

            self.metric_values["train"]["auc"].append(auc)
            self.metric_values["train"]["acc"].append(accuracy)
            self.metric_values["train"]["prec"].append(precission)
            self.metric_values["train"]["rec"].append(recall)
            self.metric_values["train"]["f1"].append(f1)
            # self.metric_values["train"]["gt_pos_rate"].append(gt_pos_rate)
            # self.metric_values["train"]["pred_pos_rate"].append(pred_pos_rate)
            self.metric_values["train"]["loss"].append(total_loss / total_examples)

            if verbose:
                ### Metrics
                self.plot_metrics("train", metrics= ["loss", "acc", "prec", "rec", "f1", "auc"])
                if self.settings["report"]["save"]:
                    plt.savefig(os.path.join(self.report_path,f'train_metrics.png'), bbox_inches='tight')

                ### Inference example - Inference
                merged_graph = self.merge_predicted_edges(copy.deepcopy(self.nxdataset["train"][-1]), predicted_edges_last_graph)
                fig = visualize_nxgraph(merged_graph, image_name = f"train {self.target_concept} inference example")
                self.metric_subplot.update_plot_with_figure(f"train {self.target_concept} inference example", fig)
                if self.target_concept == "room" or self.target_concept == "RoomWall":
                    self.cluster_rooms(merged_graph)
                if self.target_concept == "wall" or self.target_concept == "RoomWall":
                    self.cluster_walls(merged_graph)
                if self.settings["report"]["save"]:
                    plt.savefig(os.path.join(self.report_path,f'train {self.target_concept} cluster example.png'), bbox_inches='tight')

            val_loss = self.validate("val", verbose)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                self.best_model = copy.deepcopy(self.model)
                self.save_best_model()
            else:
                trigger_times += 1
            if trigger_times >= patience:
                print(f"GNNWrapper: ", Fore.BLUE + "Early stopping triggered" + Fore.WHITE)
                test_loss = self.validate( "test", verbose= True)
                break

            self.save_best_model()

        test_loss = self.validate( "test", verbose= True)
        return test_loss


    def validate(self,tag,verbose = False):
        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]
        original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        self.model = self.model.to(self.device)
        total_loss = total_examples = 0
        gt_in_val_dataset, probs_in_val_dataset = [], []

        for i, hdata in enumerate(self.hdataset["val"]):
            total_loss = total_examples = 0
            gt = hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label.to(self.device)
            gt_in_val_dataset = gt_in_val_dataset + list(gt.cpu().numpy())

            with torch.no_grad():
                hdata.to(self.device)
                logits = self.model(hdata.x_dict, hdata.edge_index_dict,\
                                    hdata.edge_index_dict)
                loss = self.criterion(logits, gt)
                total_loss += float(loss) * logits.numel()
                total_examples += logits.numel()
                
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                probs_in_val_dataset = probs_in_val_dataset + [list(probs)]
                preds = np.argmax(probs, axis=1)

            if i == len(self.hdataset["val"]) - 1:
                edge_index = list(hdata[edge_types[0],edge_types[1],edge_types[2]].edge_index.cpu().numpy())
                edge_index = np.array(list(zip(edge_index[0], edge_index[1])))
                color_code = ["black", "blue", "red"]
                predicted_edges_last_graph = [(ei[0], ei[1], {"type" : original_edge_types[preds[i]],\
                                            "label": preds[i], "viz_feat": color_code[preds[i]], "linewidth":0.5 if preds[i]==0 else 1.5,\
                                            "alpha":0.3 if preds[i]==0 else 1.}) for i, ei in enumerate(edge_index)]
            
        probs_in_val_dataset = np.concatenate(probs_in_val_dataset, axis=0)
        accuracy, precission, recall, f1, auc = self.compute_metrics_from_all_predictions(gt_in_val_dataset, probs_in_val_dataset, verbose= False)

        self.metric_values[tag]["auc"].append(auc)
        self.metric_values[tag]["acc"].append(accuracy)
        self.metric_values[tag]["prec"].append(precission)
        self.metric_values[tag]["rec"].append(recall)
        self.metric_values[tag]["f1"].append(f1)
        self.metric_values[tag]["loss"].append(total_loss / total_examples)
        # self.metric_values[tag]["gt_pos_rate"].append(gt_pos_rate)
        # self.metric_values[tag]["pred_pos_rate"].append(pred_pos_rate)
        if verbose:
            ### Metrics
            self.plot_metrics(tag, metrics= ["acc", "prec", "rec", "f1", "auc", "loss"])
            if self.settings["report"]["save"]:
                plt.savefig(os.path.join(self.report_path,f'{tag} metrics.png'), bbox_inches='tight')

            ### Inference example - Inference
            merged_graph = self.merge_predicted_edges(copy.deepcopy(self.nxdataset[tag][-1]), predicted_edges_last_graph)
            fig = visualize_nxgraph(merged_graph, image_name = f"{tag} {self.target_concept} inference example")
            self.metric_subplot.update_plot_with_figure(f"{tag} {self.target_concept} inference example", fig)
            del fig
            if self.settings["report"]["save"]:
                plt.savefig(os.path.join(self.report_path,f'{tag} {self.target_concept} inference example.png'), bbox_inches='tight')

            if self.target_concept == "RoomWall":
                clusters, inferred_graph = self.cluster_RoomWall(merged_graph)

        return loss


    def infer(self, nx_data, verbose, use_gt = False):
        original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        color_code = ["black", "blue", "red"]

        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]
        self.model = self.model.to(self.device)
        hdata = from_networkxwrapper_2_heterodata(nx_data)

        with torch.no_grad():
            hdata.to(self.device)
            print(f"dbg hdata training edge_index {hdata['ws','training','ws'].edge_index}")
            logits = self.model(hdata.x_dict, hdata.edge_index_dict,\
                                    hdata.edge_index_dict)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            edge_index = list(hdata[edge_types[0],edge_types[1],edge_types[2]].edge_index.cpu().numpy())
            edge_index = np.array(list(zip(edge_index[0], edge_index[1])))
            if use_gt:
                preds = hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label.to(self.device).cpu().numpy()
            

            predicted_edges_last_graph = [(ei[0], ei[1], {"type" : original_edge_types[preds[i]],\
                                        "label": preds[i], "viz_feat": color_code[preds[i]], "linewidth":0.5 if preds[i]==0 else 1.5,\
                                        "alpha":0.3 if preds[i]==0 else 1.}) for i, ei in enumerate(edge_index)]
            

            merged_graph = self.merge_predicted_edges(copy.deepcopy(nx_data), predicted_edges_last_graph)
            
            if verbose:
                visualize_nxgraph(merged_graph, image_name = f"Inference {self.target_concept}")
                plt.show(block=False)

            if self.target_concept == "RoomWall":
                clusters, inferred_graph = self.cluster_RoomWall(merged_graph)

        return clusters



    def infer_old(self,nx_data, verbose = False):
        gnn_settings = self.settings["gnn"]
        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]
        self.model = self.model.to(self.device)
        hdata = from_networkxwrapper_2_heterodata(nx_data)
        settings1 = self.settings["random_link_split"]
        transform = T.RandomLinkSplit(
            num_val=0.0,
            num_test=0.0,
            key= "edge_label",
            disjoint_train_ratio=0.0,
            neg_sampling_ratio=0.0,
            edge_types=edge_types,
            is_undirected = False
        )
        hdata, _, _ = transform(hdata)

        with torch.no_grad():
            hdata.to(self.device)
            preds = list(self.model(hdata.x_dict, hdata.edge_index_dict,\
                                hdata.edge_label_index_dict).cpu().numpy())

        classification_thr = gnn_settings["classification_thr"]
        self.logger.info(f"flag classification_thr {classification_thr}")

        ### Predicted edges
        edge_label_index = list(hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.cpu().numpy())
        predicted_edges = [(edge_label_index[0][i], edge_label_index[1][i], {"type" : edge_types[1], "label": preds[i],\
                                        "viz_feat": "green" if preds[i]>classification_thr else "red", "linewidth":1.5 if preds[i]>classification_thr else 1.,\
                                        "alpha":1. if preds[i]>classification_thr else 0.5, "pred": preds[i]}) for i in range(len(edge_label_index[0]))]

        if verbose:
            ### Inference example - Inference
            merged_graph = self.merge_predicted_edges(copy.deepcopy(nx_data), predicted_edges)
            # visualize_nxgraph(merged_graph, image_name = f"S-graph {self.target_concept} with predicted edges")
            if self.settings["report"]["save"]:
                plt.savefig(os.path.join(self.report_path,f'S-graph {self.target_concept} inference example.png'), bbox_inches='tight')

        if self.target_concept == "room":
            clustered_ws = self.cluster_rooms(merged_graph)
        if self.target_concept == "wall":
            clustered_ws = self.cluster_walls(merged_graph)
        if self.target_concept == "floor":
            clustered_ws = self.cluster_floors(copy.deepcopy(nx_data))

        return clustered_ws


    def compute_metrics_from_all_predictions(self, ground_truth_label, prob_label, verbose = False):

        pred_label = np.argmax(prob_label, axis=1)
        assert len(pred_label) == len(ground_truth_label)

        accuracy = accuracy_score(ground_truth_label, pred_label)
        precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth_label, pred_label, average='macro')
        conf_matrix = confusion_matrix(ground_truth_label, pred_label)

        roc_auc = roc_auc_score(ground_truth_label, prob_label, multi_class='ovr')
        
        if verbose:
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}')
            print('Confusion Matrix:\n', conf_matrix)
            print(f'ROC AUC Score: {roc_auc}')
        
        return accuracy, precision, recall, f1_score, roc_auc#, gt_pos_rate, pred_pos_rate
    

    def get_message_sharing_edges(self, nx_data):
        pass # To Be Rebuilt


    def merge_predicted_edges(self, unparented_base_graph, predictions):
        unparented_base_graph = copy.deepcopy(unparented_base_graph)
        unparented_base_graph.remove_all_edges()
        unparented_base_graph.add_edges(predictions)
        return unparented_base_graph
    

    def plot_metrics(self, tag, metrics):
            fig = plt.figure(f"{tag} Metrics")
            plt.clf()
            plt.ylim([0, 1])
            label_mapping = {"acc": "Accuracy", "prec":"Precission", "rec":"Recall", "f1":"F1", "auc":"AUC", "loss":"Loss"}
            color_mapping = {"acc": "orange", "prec":"green", "rec":"red", "f1":"purple", "auc":"brown", "loss":"blue"}
            for metric_name in metrics:
                plt.plot(np.array(self.metric_values[tag][metric_name]), label = label_mapping[metric_name], color = color_mapping[metric_name])
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Rate')
            # plt.draw()
            # plt.pause(0.001)
            self.metric_subplot.update_plot_with_figure(f"{tag} Metrics", fig)
            plt.close(fig)


    def cluster_rooms(self, old_graph):
        all_cycles = []
        graph = copy.deepcopy(old_graph)
        graph = graph.filter_graph_by_edge_attributes({"type":"ws_same_room"})
        graph.to_undirected(type= "smooth")

        def iterative_cluster_rooms(full_graph, working_graph, desired_cycle_length):
            cycles = working_graph.find_recursive_simple_cycles()
            i = 0 

            ### Filter cycles
            cycles = [frozenset(cycle) for cycle in cycles if len(cycle) == desired_cycle_length]
            cycles_unique = list(set(cycles))
            count_cycles_unique = [sum([cycle_unique == cycle for cycle in cycles]) for cycle_unique in cycles_unique]
            index = np.argsort(-np.array(count_cycles_unique))
            selected_cycles = []
            for i in index:
                if not any([any([e in final_cycle for e in cycles_unique[i]]) for final_cycle in selected_cycles]):
                    selected_cycles.append(cycles_unique[i])
                    working_graph.remove_nodes(cycles_unique[i])
            return working_graph, selected_cycles

        max_cycle_length = 10
        min_cycle_length = 2
        working_graph = copy.deepcopy(graph)
        for desired_cycle_length in reversed(range(min_cycle_length, max_cycle_length+1)):
            _, selected_cycles = iterative_cluster_rooms(graph, working_graph, desired_cycle_length)
            all_cycles += selected_cycles

        selected_rooms_dicts = []
        if all_cycles:
            viz_values = {}
            
            colors = ["cyan", "orange", "purple", "magenta", "olive", "tan", "coral", "pink", "violet", "sienna", "yellow"]
            tmp_i = 100
            for i, cycle in enumerate(all_cycles):
                room_dict = {"ws_ids": list(set(cycle))}
                room_dict["ws_centers"] = [graph.get_attributes_of_node(node_id)["center"] for node_id in list(set(cycle))]
                for node_id in cycle:
                    viz_values.update({node_id: colors[i%len(colors)]})

                if self.use_gnn_factors:
                    planes_feats_6p = [np.concatenate([graph.get_attributes_of_node(node_id)["center"],graph.get_attributes_of_node(node_id)["normal"]/np.linalg.norm(graph.get_attributes_of_node(node_id)["normal"])]) for node_id in cycle]

                    max_d = 20.
                    planes_feats_4p = [self.correct_plane_direction(plane_6_params_to_4_params(plane_feats_6p)) / np.array([1, 1, 1, max_d]) for plane_feats_6p in planes_feats_6p]
                    x = torch.cat([torch.tensor(planes_feats_4p).float(),  torch.tensor([np.zeros(len(planes_feats_4p[0]))])],dim=0).float()
                    x1, x2 = [], []
                    for i in range(x.size(0) - 1):
                        x1.append(i)
                        x2.append(x.size(0) - 1)
                    edge_index = torch.tensor(np.array([x1, x2]).astype(np.int64))
                    batch = torch.tensor(np.zeros(x.size(0)).astype(np.int64))
                    nn_outputs = self.factor_nn.infer(x, edge_index, batch, "room").numpy()[0]
                    center = np.array([nn_outputs[0], nn_outputs[1], 0]) * np.array([max_d, max_d, 1])
                else:
                    center = np.sum(np.stack([graph.get_attributes_of_node(node_id)["center"] for node_id in cycle]).astype(np.float32), axis = 0)/len(cycle)

                tmp_i += 1
                # graph.add_nodes([(tmp_i,{"type" : "room","viz_type" : "Point", "viz_data" : center,"center" : center, "viz_feat" : 'bo'})]) # TODO UNCOMMENT
                
                # for node_id in list(set(cycle)):
                #     graph.add_edges([(tmp_i, node_id, {"type": "ws_belongs_room", "x": [], "viz_feat" : 'b', "linewidth":1.0, "alpha":0.5})])


                room_dict["center"] = center
                selected_rooms_dicts.append(room_dict)
            graph.set_node_attributes("viz_feat", viz_values)
            # visualize_nxgraph(graph, image_name = "room clustering")
            if self.settings["report"]["save"]:
                plt.savefig(os.path.join(self.report_path,f'room clustering.png'), bbox_inches='tight')
                
        graph = graph.filter_graph_by_edge_attributes({"type":"ws_belongs_room"})

        return selected_rooms_dicts, graph
    
    def cluster_walls(self, graph):
        graph = copy.deepcopy(graph)
        graph = graph.filter_graph_by_edge_attributes({"type":"ws_same_wall"})
        graph.to_undirected(type= "smooth")
        
        all_edges = list(graph.get_edges_ids())
        colors = ["cyan", "orange", "purple", "magenta", "olive", "tan", "coral", "pink", "violet", "sienna", "yellow"]
        viz_values = {}
        edges_dicst = []
        tmp_i = 200
        for i, edge in enumerate(all_edges):
            wall_dict = {"ws_ids": list(set(edge))}
            wall_dict["ws_centers"] = [graph.get_attributes_of_node(node_id)["center"] for node_id in list(set(edge))]
            for node_id in edge:
                viz_values.update({node_id: colors[i%len(colors)]})
            planes_centers = np.array([np.array(graph.get_attributes_of_node(node_id)["center"]) for node_id in edge])
            
            if self.use_gnn_factors:
                max_d = 20.
                planes_centers_normalized = np.array([np.array(graph.get_attributes_of_node(node_id)["center"]) / np.array([max_d, max_d, max_d]) for node_id in edge])
                planes_feats_6p = [np.concatenate([graph.get_attributes_of_node(node_id)["center"],graph.get_attributes_of_node(node_id)["normal"]]) for node_id in edge]
                planes_feats_4p = np.array([self.correct_plane_direction(plane_6_params_to_4_params(plane_feats_6p)) / np.array([1, 1, 1, max_d]) for plane_feats_6p in planes_feats_6p])
                x = torch.cat([torch.tensor(planes_centers_normalized).float(),  torch.tensor(planes_feats_4p[:,:3])],dim=1).float()
                x = torch.cat([torch.tensor(x).float(),  torch.tensor([np.zeros(len(x[0]))])],dim=0).float()
                x1, x2 = [], []
                for i in range(x.size(0) - 1):
                    x1.append(i)
                    x2.append(x.size(0) - 1)
                edge_index = torch.tensor(np.array([x1, x2]).astype(np.int64))
                batch = torch.tensor(np.zeros(x.size(0)).astype(np.int64))
                nn_outputs = self.factor_nn.infer(x, edge_index, batch, "wall").numpy()[0]
                center = np.array([nn_outputs[0], nn_outputs[1], 0]) * np.array([max_d, max_d, 1])
            else:
                center = np.sum(np.stack([graph.get_attributes_of_node(node_id)["center"] for node_id in edge]).astype(np.float32), axis = 0)/len(edge)
                wall_points = [center, center]

            # graph.add_nodes([(tmp_i,{"type" : "wall","viz_type" : "Point", "viz_data" : center, "viz_feat" : 'bo'})]) # TODO UNCOMMENT
            tmp_i += 1
            wall_dict["center"] = center
            wall_dict["wall_points"] = planes_centers
            
            edges_dicst.append(wall_dict)
        graph.set_node_attributes("viz_feat", viz_values)
        # visualize_nxgraph(graph, image_name = "wall clustering")
        if self.settings["report"]["save"]:
            plt.savefig(os.path.join(self.report_path,f'wall clustering.png'), bbox_inches='tight')
        return edges_dicst, graph

    def cluster_floors(self, graph):
        graph = copy.deepcopy(graph)
        room_nodes_ids = graph.filter_graph_by_node_attributes({"type": "room"}).get_nodes_ids()
        graph.to_undirected(type= "smooth")
        
        floor_node_id = 500
        rooms_dicts = []
        
        if self.use_gnn_factors:
            max_d = 20.
            planes_centers = np.array([np.array(graph.get_attributes_of_node(node_id)["center"]) / np.array([max_d, max_d, max_d]) for node_id in room_nodes_ids])
            x = torch.cat([torch.tensor(planes_centers).float(),  torch.tensor([np.zeros(len(planes_centers[0]))])],dim=0).float()
            x1, x2 = [], []
            for i in range(x.size(0) - 1):
                x1.append(i)
                x2.append(x.size(0) - 1)
            edge_index = torch.tensor(np.array([x1, x2]).astype(np.int64))
            batch = torch.tensor(np.zeros(x.size(0)).astype(np.int64))
            nn_outputs = self.factor_nn.infer(x, edge_index, batch, "floor").numpy()[0]
            center = np.array([nn_outputs[0], nn_outputs[1], 0]) * np.array([max_d, max_d, max_d])
        else:
            max_d = 20.
            planes_centers = np.array([np.array(graph.get_attributes_of_node(node_id)["center"]) / np.array([max_d, max_d, max_d]) for node_id in room_nodes_ids])
            center = np.sum(np.stack([planes_center for planes_center in planes_centers]).astype(np.float32), axis = 0)/len(room_nodes_ids)
            center = np.array([center[0], center[1], 0]) * np.array([max_d, max_d, max_d])


        graph.add_nodes([(floor_node_id,{"type" : "floor","viz_type" : "Point", "viz_data" : center, "viz_feat" : 'bo'})])

        # visualize_nxgraph(graph, image_name = "floor clustering")
        
        if self.settings["report"]["save"]:
            plt.savefig(os.path.join(self.report_path,f'wall clustering.png'), bbox_inches='tight')
        return rooms_dicts
    

    def cluster_RoomWall(self, graph):
        clusters = {}
        clusters["room"], rooms_graph = self.cluster_rooms(graph)
        room_fig = visualize_nxgraph(rooms_graph, image_name = f"Inference rooms graph")
        self.metric_subplot.update_plot_with_figure("Inference rooms graph", room_fig)
        clusters["wall"], walls_graph = self.cluster_walls(graph)
        wall_fig = visualize_nxgraph(walls_graph, image_name = f"Inference walls graph")
        self.metric_subplot.update_plot_with_figure("Inference walls graph", wall_fig)
        self.metric_subplot.show(block=False)

        return clusters, graph

    
    def correct_plane_direction(self,p4):
        if p4[3] > 0:
            p4 = -1 * p4
        return p4

    def save_model(self, path = None):
        if not path:
            path = self.pth_path
        torch.save(self.model.state_dict(), path)

    def save_best_model(self, path = None):
        if not path:
            path = self.best_pth_path
            
        torch.save(self.best_model.state_dict(), path)

    def load_model(self, path = None):
        if not path:
            path = self.pth_path
        self.model.load_state_dict(torch.load(path, map_location='cpu'))