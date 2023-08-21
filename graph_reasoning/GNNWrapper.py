import numpy as np
import torch, os, sys
from torch_geometric.nn import to_hetero, GATConv
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
import tqdm
import copy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from itertools import compress
from colorama import Fore
import time
import networkx as nx
import matplotlib.pyplot as plt

graph_reasoning_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_reasoning")
sys.path.append(graph_reasoning_dir)
from graph_reasoning.from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata
graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
sys.path.append(graph_datasets_dir)
from graph_datasets.graph_visualizer import visualize_nxgraph

class GNNWrapper():
    def __init__(self, settings, report_path, logger = None) -> None:
        print(f"GNNWrapper: ", Fore.BLUE + "Initializing" + Fore.WHITE)
        # self.dataset = dataset
        self.settings = settings
        self.report_path = report_path
        self.logger = logger
        # self.writer = SummaryWriter()
        # self.hdata_loaders = self.preprocess_nxdataset(dataset)
        if self.settings["gnn"]["use_cuda"]:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f"GNNWrapper: ", Fore.BLUE + f"torch device => {self.device}" + Fore.WHITE)
        metric_values_dict = {"loss" : [], "auc" : [], "acc" : [], "prec" : [], "rec" : [], "f1" : [], "pred_pos_rate" : [], "gt_pos_rate": []}
        self.metric_values = {"train" : copy.deepcopy(metric_values_dict), "val" : copy.deepcopy(metric_values_dict),\
                            "test" : copy.deepcopy(metric_values_dict), "inference" : copy.deepcopy(metric_values_dict),}
        
    def set_dataset(self, dataset):
        self.dataset = dataset
        self.hdata_loaders = self.preprocess_nxdataset(dataset)

    def preprocess_nxdataset(self, nxdataset):
        settings1 = self.settings["random_link_split"]
        settings2 = self.settings["link_neighbor_loader"]
        normalize_features = T.NormalizeFeatures()
        edge_types = tuple(self.settings["hdata"]["edges"][0])
        rev_edge_types = tuple(self.settings["hdata"]["edges"][1])
        loaders = {}
        for tag in nxdataset.keys():
            loaders_tmp = []
            for nx_data in nxdataset[tag]:

                hdata = from_networkxwrapper_2_heterodata(nx_data)
                transform = T.RandomLinkSplit(
                    num_val=0.0,
                    num_test=0.0,
                    key= "edge_label",
                    disjoint_train_ratio=settings1["disjoint_train_ratio"],
                    neg_sampling_ratio=settings1["neg_sampling_ratio"],
                    add_negative_train_samples= settings1["add_negative_train_samples"],
                    edge_types=edge_types,
                    rev_edge_types=tuple(rev_edge_types),
                    is_undirected = settings1["is_undirected"]
                )

                hdata, _, _ = transform(hdata)
                hdata = normalize_features(hdata)

                loaders_tmp.append( LinkNeighborLoader(
                    data=hdata,
                    num_neighbors=settings2["num_neighbors"],
                    neg_sampling_ratio=0.0,
                    neg_sampling = None,
                    edge_label_index=(tuple(self.settings["hdata"]["edges"][0]), hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
                    edge_label=hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label,
                    batch_size=settings2["batch_size"],
                    shuffle=settings2["shuffle"],
                    directed = True
                ))

            loaders[tag] = loaders_tmp

        # ### Plots
        # for tag in ["train", "val"]:
        #     last_graph = copy.deepcopy(self.dataset[tag][-1])
        #     edge_index = loaders[tag][-1].data[edge_types[0],edge_types[1],edge_types[2]].edge_index.cpu().numpy()
        #     edge_label_index = loaders[tag][-1].data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.cpu().numpy()

        #     ### Message passing
        #     edge_index_tuples = [(edge_index[0][i], edge_index[1][i]) for i in range(len(edge_index[0]))]
        #     mp_edges_last_graph = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "brown", "linewidth":1.0, "alpha":0.8}) for i,pair in enumerate(edge_index_tuples)]
        #     merged_graph = self.merge_predicted_edges(copy.deepcopy(last_graph), mp_edges_last_graph)
        #     visualize_nxgraph(merged_graph, image_name = f"{tag} inference example - message passing")

        #     if self.settings["report"]["save"]:
        #         plt.savefig(os.path.join(self.report_path,f'{tag}_inference_example-mp.png'), bbox_inches='tight')

        #     ### Ground truth
        #     masked_ground_truth_in_loader = []
        #     max_value_in_edge_label = max(loaders[tag][-1].data[edge_types[0],edge_types[1],edge_types[2]].edge_label)
        #     input_id_in_samples = []
        #     for sampled_data in loaders[tag][-1]:
        #         masked_ground_truth = torch.Tensor([1 if v == max_value_in_edge_label else 0 \
        #                         for v in sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label]).cpu()
        #         masked_ground_truth_in_loader = masked_ground_truth_in_loader + list(masked_ground_truth.numpy())
        #         input_id_in_samples = input_id_in_samples + list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].input_id.cpu().numpy())

        #     classification_thr = self.settings["gnn"]["classification_thr"]
        #     predicted_edges_last_graph = [(edge_label_index[0][j], edge_label_index[1][j], {"type" : edge_types[1], "label": masked_ground_truth_in_loader[i],\
        #                                 "viz_feat": "green" if masked_ground_truth_in_loader[i]>classification_thr else "red", "linewidth":1.5 if masked_ground_truth_in_loader[i]>classification_thr else 1.,\
        #                                      "alpha":1. if masked_ground_truth_in_loader[i]>classification_thr else 0.5}) for i, j in enumerate(input_id_in_samples)]
        #     merged_graph = self.merge_predicted_edges(copy.deepcopy(last_graph), predicted_edges_last_graph)
        #     visualize_nxgraph(merged_graph, image_name = f"{tag} inference example - ground truth")
        #     if self.settings["report"]["save"]:
        #         plt.savefig(os.path.join(self.report_path,f'{tag}_inference_example-ground_truth.png'), bbox_inches='tight')

        return loaders

    def define_GCN(self):
        print(f"GNNWrapper: ", Fore.BLUE + "Defining GCN" + Fore.WHITE)

        class GNNEncoder(torch.nn.Module):
            def __init__(self, settings, in_channels_nodes, in_channels_edges):
                super().__init__()
                heads = settings["nodes"]["heads"]
                dropout = settings["nodes"]["dropout"]
                nodes_hidden_channels = settings["nodes"]["hidden_channels"]
                edges_hidden_channels = settings["edges"]["hidden_channels"]
                
                ### Nodes
                self.nodes_GATConv1 = GATConv(in_channels_nodes, nodes_hidden_channels[0], heads=heads[0], dropout=dropout)
                self.nodes_GATConv2 = GATConv(nodes_hidden_channels[0]*heads[0], nodes_hidden_channels[1], concat=False,
                                heads=heads[1], dropout=dropout)
                
                ### Edges
                self.edges_lin1 = torch.nn.Linear(in_channels_edges, edges_hidden_channels[0])
                self.edges_lin2 = torch.nn.Linear(edges_hidden_channels[0], edges_hidden_channels[1])
                
            def forward(self, x_dict, edge_index, edge_weight, edge_attr):
                ### Data gathering
                key = "ws"
                row, col = edge_index[key]
                dir_edge_keys, dir_edge_keys = edge_attr.keys()[1], edge_attr.keys()[2]
                # print(f"flag row {row}")
                # print(f"flag x {x}")
                # print(f"flag x[key][row] {x[key][row].cpu().numpy().shape()}")
                # print(f"flag x[key][col] {x[key][col].cpu().numpy().shape()}")
                # print(f"flag edge_attr {edge_attr.cpu().numpy().shape()}")
                # asdf

                ### Network forward
                # x = F.dropout(x, p=0.6, training=self.training)
                x1 = F.elu(self.nodes_GATConv1(x_dict, edge_index, edge_attr= edge_attr))
                # node_edge_attr = torch.cat([x[key][row], x[key][col], edge_attr], dim=-1)
                # edge_attr1 = copy.deepcopy(edge_attr)
                # edge_attr1[dir_edge_keys[0], dir_edge_keys[1], dir_edge_keys[2]] = torch.cat([x_dict[key][row], x_dict[key][col], edge_attr1[dir_edge_keys[0], dir_edge_keys[1], dir_edge_keys[2]]], dim=-1)
                edge_attr1 = self.edges_lin1(edge_attr) ### TODO Include node features, ELU?
                # x = F.dropout(x, p=0.6, training=self.training)
                x2 = self.nodes_GATConv2(x1, edge_index, edge_attr= edge_attr1)
                # node_edge_attr1 = torch.cat([x1[key][row], x1[key][col], edge_attr1], dim=-1)
                edge_attr2 = self.edges_lin2(edge_attr1)
                return x2, edge_attr2, edge_attr1

        class EdgeDecoder(torch.nn.Module):
            def __init__(self, settings, in_channels):
                super().__init__()
                hidden_channels = settings["hidden_channels"]
                self.decoder_lin1 = torch.nn.Linear(in_channels, hidden_channels[0])
                self.decoder_lin2 = torch.nn.Linear(hidden_channels[0], hidden_channels[1])
                self.decoder_lin3 = torch.nn.Linear(hidden_channels[1], 1)

            def forward(self, z_dict, z_emb_dict, edge_index_dict, edge_label_index):
                ### Data gathering
                row, col = edge_label_index
                key = "ws"
                e_keys = ["ws", "ws_same_room", "ws"]
                edge_index = copy.copy(edge_index_dict[e_keys[0], e_keys[1], e_keys[2]]).cpu().numpy()
                edge_index_tuples = np.array(list(zip(edge_index[0], edge_index[1])))
                edge_label_index = copy.copy(edge_label_index).cpu().numpy()
                edge_label_index_tuples = np.array(list(zip(edge_label_index[0], edge_label_index[1])))
                edge_index_to_edge_label_index = [np.argwhere((edge_label_index_single == edge_index_tuples).all(1))[0][0] for edge_label_index_single in edge_label_index_tuples]

                ### Network forward
                z = torch.cat([z_dict[key][row], z_dict[key][col], z_emb_dict[e_keys[0],e_keys[1], e_keys[2]][edge_index_to_edge_label_index]], dim=-1) ### ONLY NODE AND EDGE EMBEDDINGS
                # z = torch.cat([z_dict[key][row], z_dict[key][col]], dim=-1) ### ONLY NODE EMBEDDINGS
                # z = z_emb_dict[e_keys[0],e_keys[1], e_keys[2]][edge_index_to_edge_label_index] ### ONLY EDGE EMBEDDINGS
                z = self.decoder_lin1(z).relu()
                z = self.decoder_lin2(z).relu()
                z = self.decoder_lin3(z)
                # return F.log_softmax(x, dim=1) ### TODO TEst
                return z.view(-1)

            
        class Model(torch.nn.Module):
            def __init__(self, settings, logger):
                super().__init__()
                self.logger = logger
                # in_channels = hdata_loader
                in_channels_nodes = 5
                in_channels_edges = 1
                in_channels_decoder = 8*2 + 8
                self.encoder = GNNEncoder(settings["gnn"]["encoder"], in_channels_nodes = in_channels_nodes, in_channels_edges= in_channels_edges)
                metadata = (settings["hdata"]["nodes"], [tuple(settings["hdata"]["edges"][0]),tuple(settings["hdata"]["edges"][1])])
                self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
                self.decoder = EdgeDecoder(settings["gnn"]["decoder"], in_channels_decoder)

            def forward(self, x_dict, edge_index_dict, edge_label_index):
                z_dict, z_emb_dict, edge_attr1 = self.encoder(x_dict, edge_index = edge_index_dict, edge_weight = None, edge_attr = x_dict)
                x = self.decoder(z_dict, z_emb_dict, edge_index_dict, edge_label_index)
                return x

        self.model = Model(self.settings, self.logger)
        # self.writer.add_graph(self.model)
        # self.writer.close()

    def train(self, verbose = False):
        print(f"GNNWrapper: ", Fore.BLUE + "Training" + Fore.WHITE)
        gnn_settings = self.settings["gnn"]
        training_settings = self.settings["training"]
        self.pth_path = os.path.join(self.report_path,'model.pth')
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings["gnn"]["lr"])
        edge_types = tuple(self.settings["hdata"]["edges"][0])
        preds_in_train_dataset = []
        ground_truth_in_train_dataset = []
        # self.validate("val")
        if verbose:
            plt.show(block=False)

        for epoch in (pbar := tqdm.tqdm(range(1, training_settings["epochs"]), colour="blue")):
            pbar.set_description(f"Epoch")
            total_loss = total_examples = 0

            for i, hdata_train_graph_loader in enumerate(self.hdata_loaders["train"]):
                preds_in_loader = []
                masked_ground_truth_in_loader = []
                self.model = self.model.to(self.device)
                max_value_in_edge_label = max(hdata_train_graph_loader.data[edge_types[0],edge_types[1],edge_types[2]].edge_label)
                input_id_in_samples = []
                for sampled_data in hdata_train_graph_loader:
                    self.optimizer.zero_grad()
                    masked_ground_truth = torch.Tensor([1 if v == max_value_in_edge_label else 0 \
                                   for v in sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label]).to(self.device)
                    
                    sampled_data.to(self.device)
                    pred = self.model(sampled_data.x_dict, sampled_data.edge_index_dict,\
                                    sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index)
                    loss = F.binary_cross_entropy_with_logits(pred, masked_ground_truth)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += float(loss) * pred.numel()
                    total_examples += pred.numel()
                    preds_in_loader = preds_in_loader + list(pred.detach().cpu().numpy())

                    masked_ground_truth_in_loader = masked_ground_truth_in_loader + list(masked_ground_truth.cpu().numpy())
                    input_id_in_samples = input_id_in_samples + list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].input_id.cpu().numpy())

                preds_in_train_dataset = preds_in_train_dataset + preds_in_loader
                ground_truth_in_loader = list(masked_ground_truth_in_loader)
                ground_truth_in_train_dataset = ground_truth_in_train_dataset + ground_truth_in_loader

                if i == len(self.hdata_loaders["train"]) - 1:
                    classification_thr = gnn_settings["classification_thr"]
                    ### Predicted edges
                    edge_label_index = list(hdata_train_graph_loader.data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.cpu().numpy())

                    predicted_edges_last_graph = [(edge_label_index[0][j], edge_label_index[1][j], {"type" : edge_types[1], "label": preds_in_loader[i],\
                                                "viz_feat": "green" if preds_in_loader[i]>classification_thr else "red", "linewidth":1.5 if preds_in_loader[i]>classification_thr else 1.,\
                                             "alpha":1. if preds_in_loader[i]>classification_thr else 0.5}) for i, j in enumerate(input_id_in_samples)]

            auc = roc_auc_score(ground_truth_in_train_dataset, preds_in_train_dataset)
            accuracy, precission, recall, f1, auc, gt_pos_rate, pred_pos_rate = self.compute_metrics_from_all_predictions(ground_truth_in_train_dataset, preds_in_train_dataset, verbose= False)

            self.metric_values["train"]["auc"].append(auc)
            self.metric_values["train"]["acc"].append(accuracy)
            self.metric_values["train"]["prec"].append(precission)
            self.metric_values["train"]["rec"].append(recall)
            self.metric_values["train"]["f1"].append(f1)
            self.metric_values["train"]["gt_pos_rate"].append(gt_pos_rate)
            self.metric_values["train"]["pred_pos_rate"].append(pred_pos_rate)
            self.metric_values["train"]["loss"].append(total_loss / total_examples)

            if verbose:
                ### Metrics
                self.plot_metrics("train", metrics= ["loss", "acc", "prec", "rec", "f1", "auc"])
                if self.settings["report"]["save"]:
                    plt.savefig(os.path.join(self.report_path,f'train_metrics.png'), bbox_inches='tight')

                ### Inference example - Inference
                merged_graph = self.merge_predicted_edges(copy.deepcopy(self.dataset["train"][-1]), predicted_edges_last_graph)
                visualize_nxgraph(merged_graph, image_name = f"train inference example - inference")
                self.cluster_rooms(merged_graph)
                if self.settings["report"]["save"]:
                    plt.savefig(os.path.join(self.report_path,f'train_inference_example-inference.png'), bbox_inches='tight')

            self.validate("val", verbose)
            self.save_model()
        self.validate( "test", verbose= True)


    def validate(self,tag,verbose = False):
        hdata_loaders = self.hdata_loaders[tag]
        preds_in_val_dataset = []
        ground_truth_in_val_dataset = []
        # mp_index_tuples = []
        gnn_settings = self.settings["gnn"]
        edge_types = tuple(self.settings["hdata"]["edges"][0])
        self.model = self.model.to(self.device)
        for i, hdata_val_graph_loader in enumerate(hdata_loaders):
            preds_in_loader = []
            masked_ground_truth_in_loader = []
            max_value_in_edge_label = max(hdata_val_graph_loader.data[edge_types[0],edge_types[1],edge_types[2]].edge_label)
            input_id_in_samples = []

            for sampled_data in hdata_val_graph_loader:
                with torch.no_grad():
                    sampled_data.to(self.device)
                    preds_in_sampled = list(self.model(sampled_data.x_dict, sampled_data.edge_index_dict,\
                                        sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index).cpu().numpy())
                    preds_in_loader = preds_in_loader + preds_in_sampled
                    masked_ground_truth = torch.Tensor([1 if v == max_value_in_edge_label else 0 \
                                   for v in sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label]).to(self.device)
                    masked_ground_truth_in_loader = masked_ground_truth_in_loader + list(masked_ground_truth.cpu().numpy())
                    input_id_in_samples = input_id_in_samples + list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].input_id.cpu().numpy())
            preds_in_val_dataset = preds_in_val_dataset + preds_in_loader
            ground_truth_in_loader = list(masked_ground_truth_in_loader)
            ground_truth_in_val_dataset = ground_truth_in_val_dataset + ground_truth_in_loader

            if i == len(hdata_loaders) - 1:
                classification_thr = gnn_settings["classification_thr"]
                ### Predicted edges
                edge_label_index = list(hdata_val_graph_loader.data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.cpu().numpy())
                predicted_edges_last_graph = [(edge_label_index[0][j], edge_label_index[1][j], {"type" : edge_types[1], "label": preds_in_loader[i],\
                                             "viz_feat": "green" if preds_in_loader[i]>classification_thr else "red", "linewidth":1.5 if preds_in_loader[i]>classification_thr else 1.,\
                                             "alpha":1. if preds_in_loader[i]>classification_thr else 0.5}) for i, j in enumerate(input_id_in_samples)]
        auc = roc_auc_score(ground_truth_in_val_dataset, preds_in_val_dataset)
        accuracy, precission, recall, f1, auc, gt_pos_rate, pred_pos_rate = self.compute_metrics_from_all_predictions(ground_truth_in_val_dataset, preds_in_val_dataset, verbose= False)

        self.metric_values[tag]["auc"].append(auc)
        self.metric_values[tag]["acc"].append(accuracy)
        self.metric_values[tag]["prec"].append(precission)
        self.metric_values[tag]["rec"].append(recall)
        self.metric_values[tag]["f1"].append(f1)
        self.metric_values[tag]["gt_pos_rate"].append(gt_pos_rate)
        self.metric_values[tag]["pred_pos_rate"].append(pred_pos_rate)
        if verbose:
            ### Metrics
            self.plot_metrics(tag, metrics= ["acc", "prec", "rec", "f1", "auc"])
            if self.settings["report"]["save"]:
                plt.savefig(os.path.join(self.report_path,f'{tag}_metrics.png'), bbox_inches='tight')

            ### Inference example - Inference
            merged_graph = self.merge_predicted_edges(copy.deepcopy(self.dataset[tag][-1]), predicted_edges_last_graph)
            visualize_nxgraph(merged_graph, image_name = f"{tag} inference example - inference")
            if self.settings["report"]["save"]:
                plt.savefig(os.path.join(self.report_path,f'{tag}_inference_example-inference.png'), bbox_inches='tight')

    
    def infer(self,nx_data,verbose = False):
        # mp_index_tuples = []
        device = "cpu"
        # self.pth_path = '/home/adminpc/reasoning_ws/src/graph_reasoning/pths/model.pth'
        gnn_settings = self.settings["gnn"]
        edge_types = tuple(self.settings["hdata"]["edges"][0])
        rev_edge_types = tuple(self.settings["hdata"]["edges"][1])
        self.model = self.model.to(device)
        hdata = from_networkxwrapper_2_heterodata(nx_data)
        transform = T.RandomLinkSplit(
            num_val=0.0,
            num_test=0.0,
            key= "edge_label",
            disjoint_train_ratio=0.0,
            neg_sampling_ratio=0.0,
            add_negative_train_samples= False,
            edge_types=edge_types,
            rev_edge_types=tuple(rev_edge_types),
            is_undirected = False
        )
        hdata, _, _ = transform(hdata)
        hdata = T.NormalizeFeatures()(hdata)

        with torch.no_grad():
            hdata.to(device)
            preds = list(self.model(hdata.x_dict, hdata.edge_index_dict,\
                                hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label_index).cpu().numpy())

        classification_thr = gnn_settings["classification_thr"]
        ### Predicted edges
        edge_label_index = list(hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.cpu().numpy())
        predicted_edges = [(edge_label_index[0][i], edge_label_index[1][i], {"type" : edge_types[1], "label": preds[i],\
                                        "viz_feat": "green" if preds[i]>classification_thr else "red", "linewidth":1.5 if preds[i]>classification_thr else 1.,\
                                        "alpha":1. if preds[i]>classification_thr else 0.5}) for i in range(len(edge_label_index[0]))]

        if verbose:
            ### Inference example - Inference
            merged_graph = self.merge_predicted_edges(copy.deepcopy(nx_data), predicted_edges)
            if self.settings["report"]["save"]:
                plt.savefig(os.path.join(self.report_path,'inference from s_graph.png'), bbox_inches='tight')

        clustered_ws = self.cluster_rooms(merged_graph)
        # # cluster_dict = {}
        # # for i in clustered_ws:
        # return cluster_dict

        return clustered_ws


    def compute_metrics_from_all_predictions(self, ground_truth_label, pred_label, verbose = False):

        assert len(pred_label) == len(ground_truth_label)
        classification_thr = self.settings["gnn"]["classification_thr"]
        pred_onehot_label = np.where(np.array(pred_label) > classification_thr, 1, 0)

        len_all_indexes = len(pred_label)
        len_predicted_positives = sum(pred_onehot_label)
        len_predicted_negatives = len_all_indexes - len_predicted_positives
        len_actual_positives = sum(ground_truth_label)
        len_actual_negatives = len_all_indexes - len_actual_positives

        pred_pos_rate = len_predicted_positives / len_all_indexes
        gt_pos_rate = len_actual_positives / len_all_indexes

        auc = roc_auc_score(ground_truth_label, pred_label)

        true_positives = sum(compress(np.array(pred_onehot_label) == np.array(ground_truth_label), [True if n==1. else False for n in pred_onehot_label]))
        false_positives = len_predicted_positives - true_positives
        false_negatives = len_actual_positives - true_positives
        true_negatives = len_predicted_negatives - false_negatives
        confusion_matrix = [[true_positives, false_positives], [false_negatives, true_negatives]]

        if verbose:
            print("=== Confusion Matrix ===")
            print(f"TP: {true_positives:.3f} | FP: {false_positives:.3f}")
            print("------------------------")
            print(f"FN: {false_negatives:.3f} | TN: {true_negatives:.3f}")

        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) 
        precission = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.
        f1 = 2*precission*recall / (precission + recall) if precission and recall else 0.
        
        if verbose:
            print("======= Metics =========")
            print(f"GT positives rate {gt_pos_rate:.3f}, predicted positives rate {pred_pos_rate:.3f}")
            print(f"Accuracy {accuracy:.3f}, Precission {precission:.3f}, Recall {recall:.3f}, F1 {f1:.3f}, AUC {auc:.3f}")
        
        return accuracy, precission, recall, f1, auc, gt_pos_rate, pred_pos_rate
    

    # def get_message_sharing_edges(self, nx_data):
    #     device = "cpu"
    #     edge_types = tuple(self.settings["hdata"]["edges"][0])
    #     mp_edges = {}
    #     label_edges = {}

    #     loader = self.preprocess_data([nx_data], "train")

    #     for tag in ["train", "val", "test"]:
    #         edge_label_full = []
    #         edge_label_index_full = []
    #         mp_edge_index_full = []

    #         for sampled_data in loader[tag][0]:
    #             sampled_data.to(device)
    #             mp_edge_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_index.numpy()
    #             mp_edge_index_full = mp_edge_index_full + [(indexes[0], indexes[1]) for indexes in zip(mp_edge_index[0], mp_edge_index[1])]
    #             edge_label_full = edge_label_full + list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.numpy())
    #             edge_label_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.numpy()
    #             edge_label_index_full = edge_label_index_full + [(indexes[0], indexes[1]) for indexes in zip(edge_label_index[0], edge_label_index[1])]

    #         mp_edges[tag] = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "grey"}) for i,pair in enumerate(mp_edge_index_full)]
    #         label_edges[tag] = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "g" if edge_label_full[i] == 1 else "r"}) for i,pair in enumerate(edge_label_index_full)]

    #     sampled_data = self.preprocess_data(nx_data, "inference")
    #     sampled_data.to(device)
    #     mp_edge_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_index.numpy()
    #     mp_edge_index = [(indexes[0], indexes[1]) for indexes in zip(mp_edge_index[0], mp_edge_index[1])]
    #     edge_label = list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.numpy())
    #     edge_label_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.numpy()
    #     edge_label_index = [(indexes[0], indexes[1]) for indexes in zip(edge_label_index[0], edge_label_index[1])]

    #     mp_edges["inference"] = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "grey"}) for i,pair in enumerate(mp_edge_index)]
    #     label_edges["inference"] = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "g" if edge_label[i] == 1 else "r"}) for i,pair in enumerate(edge_label_index)]
        
    #     return mp_edges, label_edges


    def merge_predicted_edges(self, unparented_base_graph, predictions):
        unparented_base_graph = copy.deepcopy(unparented_base_graph)
        unparented_base_graph.remove_all_edges()
        unparented_base_graph.add_edges(predictions)
        return unparented_base_graph
    

    def plot_metrics(self, tag, metrics):
            plt.figure(f"{tag} Metrics")
            plt.clf()
            plt.ylim([0, 1])
            label_mapping = {"acc": "Accuracy", "prec":"Precission", "rec":"Recall", "f1":"F1", "auc":"AUC", "loss":"Loss"}
            color_mapping = {"acc": "orange", "prec":"green", "rec":"red", "f1":"purple", "auc":"brown", "loss":"blue"}
            for metric_name in metrics:
                plt.plot(np.array(self.metric_values[tag][metric_name]), label = label_mapping[metric_name], color = color_mapping[metric_name])
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Rate')
            plt.draw()
            plt.pause(0.001)


    def cluster_rooms(self, graph):
        selected_cycles = []
        graph = copy.deepcopy(graph)
        graph = graph.filter_graph_by_edge_attributes({"viz_feat": "green"})
        graph.to_undirected()
        cycles = graph.find_recursive_simple_cycles()
        colors = ["cyan", "yellow", "orange", "purple", "magenta", "olive", "tan", "coral", "pink", "violet", "sienna"]
        viz_values = {}
        i = 0 

        ### Filter cycles
        cycles = [frozenset(cycle) for cycle in cycles]
        cycles_unique = list(set(cycles))
        count_cycles_unique = [sum([cycle_unique == cycle for cycle in cycles]) for cycle_unique in cycles_unique]
        index = np.argsort(-np.array(count_cycles_unique))
        final_cycles = []
        for i in index:
            if not any([any([e in final_cycle for e in cycles_unique[i]]) for final_cycle in final_cycles]):
                final_cycles.append(cycles_unique[i])



        if final_cycles:
            for cycle in final_cycles:
                if len(cycle) == 4:
                    room_dict = {"ws_ids": cycle}
                    i += 1
                    for node_id in cycle:
                        viz_values.update({node_id: colors[i%len(colors)]})
                    center = sum(np.stack([graph.get_attributes_of_node(node_id)["center"] for node_id in cycle]).astype(np.float32))/len(cycle)
                    room_dict["center"] = center
                    selected_cycles.append(room_dict)
            graph.set_node_attributes("viz_feat", viz_values)
            visualize_nxgraph(graph, image_name = "only positive predictions")
        return selected_cycles
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.pth_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.pth_path))