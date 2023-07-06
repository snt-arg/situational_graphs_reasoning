import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import GCNConv, to_hetero, SAGEConv
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import HeteroData
from from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata
import tqdm
import copy
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from itertools import compress
from colorama import Fore, Back, Style

import time

import networkx as nx
import matplotlib.pyplot as plt


class GNNWrapper():
    def __init__(self, dataset, settings) -> None:
        print(f"GNNWrapper: ", Fore.BLUE + "Initializing" + Fore.WHITE)
        self.settings = settings
        self.hdata_loaders = self.preprocess_nxdata_v1(dataset, "train")
        # self.hdata_loaders = self.preprocess_nxdataset_v2(dataset)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = "cpu"
        print(f"GNNWrapper: ", Fore.BLUE + f"torch device => {self.device}" + Fore.WHITE)
        metric_values_dict = {"loss" : [], "auc" : [], "acc" : [], "prec" : [], "rec" : [], "f1" : [], "pred_pos_rate" : []}
        self.metric_values = {"train" : copy.deepcopy(metric_values_dict), "val" : copy.deepcopy(metric_values_dict),\
                            "test" : copy.deepcopy(metric_values_dict), "inference" : copy.deepcopy(metric_values_dict),}

    def preprocess_nxdata_v1(self, nxdata_list, stage = "train"):
        print(f"GNNWrapper: ", Fore.BLUE + "Preprocessing data" + Fore.WHITE)
        settings1 = self.settings["random_link_split"]
        settings2 = self.settings["link_neighbor_loader"]
        normalize_features = T.NormalizeFeatures()
        
        if stage == "train":
            train_loaders, val_loaders, test_loaders = [], [], []
            edge_types = tuple(settings1["edge_types"])
            for i, nx_data in enumerate(nxdata_list):
                data = from_networkxwrapper_2_heterodata(nx_data)

                transform = T.RandomLinkSplit(
                    num_val=settings1["num_val"],
                    num_test=settings1[ "num_test"],
                    key= "edge_label",
                    disjoint_train_ratio=settings1["disjoint_train_ratio"],
                    neg_sampling_ratio=settings1["neg_sampling_ratio"],
                    add_negative_train_samples= settings1["add_negative_train_samples"],
                    edge_types=edge_types,
                    rev_edge_types=tuple(settings1["rev_edge_types"]),
                    is_undirected = settings1["is_undirected"]
                )
                train_data, val_data, test_data = transform(data)
                train_data, val_data, test_data = normalize_features(train_data), normalize_features(val_data), normalize_features(test_data)

                # assert train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.min() == 1
                # assert train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.max() == 1
                    
                train_loaders.append( LinkNeighborLoader(
                    data=train_data,
                    num_neighbors=settings2["num_neighbors"],
                    neg_sampling_ratio=settings2["neg_sampling_ratio"],
                    edge_label_index=(tuple(settings1["edge_types"]), train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
                    edge_label=train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label,
                    batch_size=settings2["batch_size"],
                    shuffle=settings2["shuffle"],
                ))

                # sampled_data = next(iter(train_loaders[0]))
                # assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.min() == 0
                # assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.max() == 1

                val_loaders.append( LinkNeighborLoader(
                    data=val_data,
                    num_neighbors=settings2["num_neighbors"],
                    edge_label_index=(tuple(settings1["edge_types"]), val_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
                    edge_label=val_data[edge_types[0],edge_types[1],edge_types[2]].edge_label,
                    batch_size=settings2["batch_size"]*3,
                    shuffle=False,
                ))

                # if len(val_loaders[0]) != 0:
                #     sampled_data = next(iter(val_loaders[0]))
                    # assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.min() == 0
                    # assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.max() == 1

                test_loaders.append( LinkNeighborLoader(
                    data=test_data,
                    num_neighbors=settings2["num_neighbors"],
                    edge_label_index=(tuple(settings1["edge_types"]), test_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
                    edge_label=test_data[edge_types[0],edge_types[1],edge_types[2]].edge_label,
                    batch_size=settings2["batch_size"]*3,
                    shuffle=False,
                ))

                # if len(test_loaders[0]):
                #     sampled_data = next(iter(test_loaders[0]))
                    # assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.min() == 0
                    # assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.max() == 1


            loaders = {"train" : train_loaders, "val": val_loaders, "test": test_loaders}

        elif stage == "inference":
            hdata = from_networkxwrapper_2_heterodata(nxdata_list)
            hdata = normalize_features(hdata)
            transform = T.RandomLinkSplit(
                num_val=0.,
                num_test=0.,
                disjoint_train_ratio=0.,
                add_negative_train_samples= False,
                edge_types=tuple(settings1["edge_types"]),
            )

            train_data, val_data, test_data = transform(hdata)            
            
            edge_types = tuple(settings1["edge_types"])
            assert train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.min() == 1
            assert train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.max() == 1

            # loader =  LinkNeighborLoader(
            #     data=train_data,
            #     num_neighbors=settings2["num_neighbors"],
            #     edge_label_index=(tuple(settings1["edge_types"]), train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
            #     edge_label=train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label,
            #     batch_size=settings2["batch_size"]*3,
            #     shuffle=False,
            # )

            # loaders = loader
            loaders = train_data

        return loaders
    

    def preprocess_nxdataset_v2(self, nxdataset):
        settings1 = self.settings["random_link_split"]
        settings2 = self.settings["link_neighbor_loader"]
        normalize_features = T.NormalizeFeatures()
        edge_types = tuple(settings1["edge_types"])
        loaders = {}
        for key in nxdataset.keys():
            loaders_tmp = []
            for nx_data in nxdataset[key]:
                hdata = from_networkxwrapper_2_heterodata(nx_data)
                hdata = normalize_features(hdata)
                # num_neighbors_dict = {}
                # num_neighbors_dict[edge_types[0],edge_types[1],edge_types[2]] = settings2["num_neighbors"]
                loaders_tmp.append( LinkNeighborLoader(
                    data=hdata,
                    num_neighbors=settings2["num_neighbors"],
                    neg_sampling_ratio=0.0,
                    neg_sampling = None,
                    edge_label_index=(tuple(settings1["edge_types"]), hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
                    edge_label=hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label,
                    batch_size=settings2["batch_size"],
                    shuffle=settings2["shuffle"],
                    directed = True
                ))

            loaders[key] = loaders_tmp

        return loaders
        

    def visualize_graph(self, G, color):
        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                        node_color=color, cmap="Set2")
        plt.show()


    def visualize_embedding(self, h, color, epoch=None, loss=None):
        plt.figure(figsize=(7,7))
        plt.xticks([])
        plt.yticks([])
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
        plt.show()

    def define_GCN(self):
        print(f"GNNWrapper: ", Fore.BLUE + "Defining GCN" + Fore.WHITE)
        # settings = self.settings["gnn"]
        # class GCN(torch.nn.Module):
        #     def __init__(self, hidden_channels):
        #         super().__init__()
        #         torch.manual_seed(1234)
        #         self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        #         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        #         self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        #     def forward(self, x, edge_index):
        #         h = self.conv1(x, edge_index).relu()
        #         h = self.conv2(h, edge_index)

        #         return h
            
        class GNNEncoder(torch.nn.Module):
            def __init__(self, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = SAGEConv((-1, -1), hidden_channels)
                self.conv2 = SAGEConv((-1, -1), out_channels)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index)
                return x
            
        class EdgeDecoder(torch.nn.Module):
            def __init__(self, hidden_channels):
                super().__init__()
                self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
                self.lin2 = torch.nn.Linear(hidden_channels, 1)

            def forward(self, z_dict, edge_label_index):
                row, col = edge_label_index
                key = "ws" ### TODO from settings
                z = torch.cat([z_dict[key][row], z_dict[key][col]], dim=-1)

                z = self.lin1(z).relu()
                z = self.lin2(z)
                return z.view(-1)

            
        class Model(torch.nn.Module):
            def __init__(self, hdata_loader, hidden_channels):
                super().__init__()
                self.encoder = GNNEncoder(hidden_channels, hidden_channels)
                self.encoder = to_hetero(self.encoder, hdata_loader.data.metadata(), aggr='sum')
                self.decoder = EdgeDecoder(hidden_channels)

            def forward(self, x_dict, edge_index_dict, edge_label_index):
                z_dict = self.encoder(x_dict, edge_index_dict)
                x = self.decoder(z_dict, edge_label_index)
                # x_soft = F.log_softmax(x, dim=0)
                return x

            def reset_embeddings(self, num_nodes, num_features, hidden_channels):
                self.ws_lin = torch.nn.Linear(num_features, hidden_channels)
                self.ws_emb = torch.nn.Embedding(num_nodes, hidden_channels)

            # def forward(self, data: HeteroData) -> Tensor:
            #     x_dict = {
            #     "ws": self.ws_lin(data["ws"].x) + self.ws_emb(data["ws"].node_id),
            #     } 

            #     # `x_dict` holds feature matrices of all node types
            #     # `edge_index_dict` holds all edge indices of all edge types
            #     x_dict = self.gnn(x_dict, data.edge_index_dict)

            #     pred = self.classifier(
            #         x_dict["ws"],
            #         x_dict["ws"],
            #         data["ws", "ws_same_room", "ws"].edge_label_index,
            #     )
            #     return pred
            
        self.model = Model(self.hdata_loaders["train"][0], hidden_channels=32)


    def train(self, verbose = False):
        print(f"GNNWrapper: ", Fore.BLUE + "Training" + Fore.WHITE)
        gnn_settings = self.settings["gnn"]
        training_settings = self.settings["training"]
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings["gnn"]["lr"])
        edge_types = tuple(self.settings["random_link_split"]["edge_types"])
        preds_in_train_dataset = []
        ground_truth_in_train_dataset = []
        self.validate(self.hdata_loaders["val"], "val")
        if verbose:
            plt.show()

        for epoch in (pbar := tqdm.tqdm(range(1, training_settings["epochs"]), colour="blue")):
            pbar.set_description(f"Epoch")
            total_loss = total_examples = 0

            # for hdata_train_graph_loader in (pbar2 := tqdm.tqdm(self.hdata_loaders["train"], colour="blue", leave=False)):
            #     pbar2.set_description(f"Loader")
            for hdata_train_graph_loader in self.hdata_loaders["train"]:
                preds_in_loader = []
                masked_ground_truth_in_loader = []
                self.model.reset_embeddings(hdata_train_graph_loader.data["ws"].num_nodes , gnn_settings["num_features"], gnn_settings["hidden_channels"])
                self.model = self.model.to(self.device)
                # print(f"flag train {hdata_train_graph_loader.data[edge_types[0],edge_types[1],edge_types[2]].edge_label}")

                for sampled_data in hdata_train_graph_loader:
                    self.optimizer.zero_grad()
                    
                    sampled_data.to(self.device)
                    # pred = self.model(sampled_data)
                    pred = self.model(sampled_data.x_dict, sampled_data.edge_index_dict,\
                                    sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index)
                    max_value_in_edge_label = max(sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label)
                    masked_ground_truth = torch.Tensor([1 if v == max_value_in_edge_label else 0 \
                                   for v in sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label]).to(self.device)
                    # print(f"flag masked_ground_truth { masked_ground_truth}")
                    loss = F.binary_cross_entropy_with_logits(pred, masked_ground_truth)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += float(loss) * pred.numel()
                    total_examples += pred.numel()
                    preds_in_loader = preds_in_loader + list(pred.detach().cpu().numpy())
                    masked_ground_truth_in_loader = masked_ground_truth_in_loader + list(masked_ground_truth.cpu().numpy())

                preds_in_train_dataset = preds_in_train_dataset + preds_in_loader
                ground_truth_in_loader = list(masked_ground_truth_in_loader)
                ground_truth_in_train_dataset = ground_truth_in_train_dataset + ground_truth_in_loader

            auc = roc_auc_score(ground_truth_in_train_dataset, preds_in_train_dataset)
            accuracy, precission, recall, f1, auc, pred_pos_rate = self.compute_metrics_from_all_predictions(ground_truth_in_train_dataset, preds_in_train_dataset, verbose= False)

            self.metric_values["train"]["auc"].append(auc)
            self.metric_values["train"]["acc"].append(accuracy)
            self.metric_values["train"]["prec"].append(precission)
            self.metric_values["train"]["rec"].append(recall)
            self.metric_values["train"]["f1"].append(f1)
            self.metric_values["train"]["pred_pos_rate"].append(pred_pos_rate)

            if verbose:
                self.metric_values["train"]["loss"].append(total_loss / total_examples)
                plt.figure("Train Metrics")
                plt.clf()
                plt.ylim([0, 1])
                plt.plot(np.array(self.metric_values["train"]["loss"]), label = "Loss")
                plt.plot(np.array(self.metric_values["train"]["acc"]), label = "Accuracy")
                plt.plot(np.array(self.metric_values["train"]["prec"]), label = "Precission")
                plt.plot(np.array(self.metric_values["train"]["rec"]), label = "Recall")
                plt.plot(np.array(self.metric_values["train"]["f1"]), label = "F1")
                plt.plot(np.array(self.metric_values["train"]["auc"]), label = "AUC")
                plt.plot(np.array(self.metric_values["train"]["pred_pos_rate"]), label = "predicted positives rate")
                plt.legend()
                plt.xlabel('Epochs')
                plt.ylabel('Rate')
                plt.draw()
                plt.pause(0.001)

            self.validate(self.hdata_loaders["val"], "val", verbose)
        self.validate(self.hdata_loaders["test"], "test", verbose= True)


    def validate(self, hdata_loaders, tag,verbose = False):
        preds_in_val_dataset = []
        ground_truth_in_val_dataset = []
        # mp_index_tuples = []
        gnn_settings = self.settings["gnn"]
        edge_types = tuple(self.settings["random_link_split"]["edge_types"])

        for hdata_val_graph_loader in hdata_loaders:
            preds_in_loader = []
            # self.model.reset_embeddings(hdata_val_graph_loader.data["ws"].num_nodes , gnn_settings["num_features"], gnn_settings["hidden_channels"]) ### TODO with loader
            # self.model.reset_embeddings(hdata_val_graph_loader["ws"].num_nodes , gnn_settings["num_features"], gnn_settings["hidden_channels"])
            self.model = self.model.to(self.device)
            max_value_in_edge_label = max(hdata_val_graph_loader.data[edge_types[0],edge_types[1],edge_types[2]].edge_label)

            masked_ground_truth = torch.Tensor([1 if v == max_value_in_edge_label else 0 \
                                   for v in hdata_val_graph_loader.data[edge_types[0],edge_types[1],edge_types[2]].edge_label]).cpu().numpy()

            
            for sampled_data in hdata_val_graph_loader:
                with torch.no_grad():                   
                    sampled_data.to(self.device)
                    # preds_in_sampled = list(self.model(sampled_data).cpu().numpy())
                    preds_in_sampled = list(self.model(sampled_data.x_dict, sampled_data.edge_index_dict,\
                                    sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index).cpu().numpy())
                    preds_in_loader = preds_in_loader + preds_in_sampled

            preds_in_val_dataset = preds_in_val_dataset + preds_in_loader
            ground_truth_in_loader = list(masked_ground_truth)
            ground_truth_in_val_dataset = ground_truth_in_val_dataset + ground_truth_in_loader

        auc = roc_auc_score(ground_truth_in_val_dataset, preds_in_val_dataset)
        accuracy, precission, recall, f1, auc, pred_pos_rate = self.compute_metrics_from_all_predictions(ground_truth_in_val_dataset, preds_in_val_dataset, verbose= False)

        self.metric_values[tag]["auc"].append(auc)
        self.metric_values[tag]["acc"].append(accuracy)
        self.metric_values[tag]["prec"].append(precission)
        self.metric_values[tag]["rec"].append(recall)
        self.metric_values[tag]["f1"].append(f1)
        self.metric_values[tag]["pred_pos_rate"].append(pred_pos_rate)

        if verbose:
            plt.figure("Val Metrics")
            plt.clf()
            plt.ylim([0, 1])
            plt.plot(np.array(self.metric_values["val"]["acc"]), label = "Accuracy")
            plt.plot(np.array(self.metric_values["val"]["prec"]), label = "Precission")
            plt.plot(np.array(self.metric_values["val"]["rec"]), label = "Recall")
            plt.plot(np.array(self.metric_values["val"]["f1"]), label = "F1")
            plt.plot(np.array(self.metric_values["val"]["auc"]), label = "AUC")
            plt.plot(np.array(self.metric_values["val"]["pred_pos_rate"]), label = "predicted positives rate")
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Rate')
            plt.draw()
            plt.pause(0.001)

    
    def infer(self, nxdataset, use_label = False):
        print(f"GNNWrapper: ", Fore.BLUE + "Final inference" + Fore.WHITE)
        device = "cpu"
        gnn_settings = self.settings["gnn"]
        rls_settings = self.settings["random_link_split"]
        # hdata_loader = self.preprocess_nxdataset_v2(nxdataset)["inference"][0]
        hdata_loader = self.preprocess_nxdata_v1(nxdataset, "train")["train"][0]
        edge_types = tuple(rls_settings["edge_types"])
        self.model = self.model.to(device)
        self.model.reset_embeddings(hdata_loader.data["ws"].num_nodes , gnn_settings["num_features"], gnn_settings["hidden_channels"])
        
        
        # edge_labels = hdata_loader.data[edge_types[0],edge_types[1],edge_types[2]].edge_label.to(device).numpy()
        # print(f"flag edge_labels {len(edge_labels)}")
        edge_label_index = hdata_loader.data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.to(device).numpy()
        edge_label_index_tuples = [(edge_label_index[0][i], edge_label_index[1][i]) for i in range(len(edge_label_index[0]))]
        edge_labels = []
        pred_labels = []
        
        for sampled_data in hdata_loader:
            with torch.no_grad():
                edge_labels = edge_labels + list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.to(device).numpy())
                sampled_data.to(device)
                pred = list(self.model(sampled_data.x_dict, sampled_data.edge_index_dict,\
                                    sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index).numpy())
                print(f"flag pred {len(pred)}")
                pred_labels = pred_labels + pred

        if use_label:
            max_value_in_edge_label = max(edge_labels)
            masked_ground_truth = torch.Tensor([1 if v == max_value_in_edge_label else 0 \
                            for v in edge_labels]).to(device)
            print(f"flag masked_ground_truth {len(masked_ground_truth)}")
            print(f"flag pred_labels {len(pred_labels)}")
            auc = roc_auc_score(masked_ground_truth, pred_labels)
            accuracy, precission, recall, f1, auc, pred_pos_rate  = self.compute_metrics_from_all_predictions(masked_ground_truth, pred_labels, verbose= True)

        predicted_edges = []
        for i,pair in enumerate(edge_label_index_tuples):
            predicted_edges.append((pair[0], pair[1], {"type" : edge_types[1], "label": pred_labels[i], "viz_feat": "green" if pred_labels[i]==1 else "red"}))

        return predicted_edges

    def compute_metrics_from_all_predictions(self, ground_truth_label, pred_label, verbose = False):

        assert len(pred_label) == len(ground_truth_label)
        pred_onehot_label = np.where(np.array(pred_label) > 0.5, 1, 0)

        len_all_indexes = len(pred_label)
        len_predicted_positives = sum(pred_onehot_label)
        len_predicted_negatives = len_all_indexes - len_predicted_positives
        len_actual_positives = sum(ground_truth_label)
        len_actual_negatives = len_all_indexes - len_actual_positives

        pred_pos_rate = len_predicted_positives / len_all_indexes
        act_pos_rate = len_actual_positives / len_all_indexes

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
            print(f"predicted positives rate {pred_pos_rate:.3f}, actual positives rate {act_pos_rate:.3f}")
            print(f"Accuracy {accuracy:.3f}, Precission {precission:.3f}, Recall {recall:.3f}, F1 {f1:.3f}, AUC {auc:.3f}")
        
        return accuracy, precission, recall, f1, auc, pred_pos_rate
    

    def get_message_sharing_edges(self, nx_data):
        device = "cpu"
        rls_settings = self.settings["random_link_split"]
        edge_types = tuple(rls_settings["edge_types"])
        mp_edges = {}
        label_edges = {}

        loader = self.preprocess_data([nx_data], "train")

        for tag in ["train", "val", "test"]:
            edge_label_full = []
            edge_label_index_full = []
            mp_edge_index_full = []

            for sampled_data in loader[tag][0]:
                sampled_data.to(device)
                mp_edge_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_index.numpy()
                mp_edge_index_full = mp_edge_index_full + [(indexes[0], indexes[1]) for indexes in zip(mp_edge_index[0], mp_edge_index[1])]
                edge_label_full = edge_label_full + list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.numpy())
                edge_label_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.numpy()
                edge_label_index_full = edge_label_index_full + [(indexes[0], indexes[1]) for indexes in zip(edge_label_index[0], edge_label_index[1])]

            mp_edges[tag] = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "grey"}) for i,pair in enumerate(mp_edge_index_full)]
            label_edges[tag] = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "g" if edge_label_full[i] == 1 else "r"}) for i,pair in enumerate(edge_label_index_full)]

        sampled_data = self.preprocess_data(nx_data, "inference")
        sampled_data.to(device)
        mp_edge_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_index.numpy()
        mp_edge_index = [(indexes[0], indexes[1]) for indexes in zip(mp_edge_index[0], mp_edge_index[1])]
        edge_label = list(sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.numpy())
        edge_label_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.numpy()
        edge_label_index = [(indexes[0], indexes[1]) for indexes in zip(edge_label_index[0], edge_label_index[1])]

        mp_edges["inference"] = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "grey"}) for i,pair in enumerate(mp_edge_index)]
        label_edges["inference"] = [(pair[0], pair[1], {"type" : edge_types[1], "viz_feat": "g" if edge_label[i] == 1 else "r"}) for i,pair in enumerate(edge_label_index)]
        
        return mp_edges, label_edges
