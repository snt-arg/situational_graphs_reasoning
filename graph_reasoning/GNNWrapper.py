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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from itertools import compress

import time

import networkx as nx
import matplotlib.pyplot as plt


class GNNWrapper():
    def __init__(self, nx_data_full, settings) -> None:
        self.nx_data_full = nx_data_full
        self.settings = settings
        self.hdata_loaders = self.preprocess_data(nx_data_full, "train")
        print(f"GNNWrapper: Initializing")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"GNNWrapper: torch device => {self.device}")
        self.plot_values = {"train" : {"loss" : []}, "val" : {"auc" : [], "acc" : [], "prec" : [], "rec" : [], "f1" : []}}

    def preprocess_data(self, nxdata_list, stage = "train"):
        print(f"GNNWrapper: Preprocessing data")
        settings1 = self.settings["random_link_split"]
        settings2 = self.settings["link_neighor_loader"]
        
        if stage == "train":
            train_loaders, val_loaders, test_loaders = [], [], []
            for i, nx_data in enumerate(nxdata_list):
                data = from_networkxwrapper_2_heterodata(nx_data)
                transform = T.RandomLinkSplit(
                    num_val=settings1["num_val"],
                    num_test=settings1[ "num_test"],
                    disjoint_train_ratio=settings1["disjoint_train_ratio"],
                    neg_sampling_ratio=settings1["neg_sampling_ratio"],
                    add_negative_train_samples= settings1["add_negative_train_samples"],
                    edge_types=tuple(settings1["edge_types"]),
                    # rev_edge_types=tuple(settings1["rev_edge_types"]), 
                )
                train_data, val_data, test_data = transform(data)

                edge_types = tuple(settings1["edge_types"])
                assert train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.min() == 1
                assert train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.max() == 1
                    
                train_loaders.append( LinkNeighborLoader(
                    data=train_data,
                    num_neighbors=settings2["num_neighbors"],
                    neg_sampling_ratio=settings2["neg_sampling_ratio"],
                    edge_label_index=(tuple(settings1["edge_types"]), train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
                    edge_label=train_data[edge_types[0],edge_types[1],edge_types[2]].edge_label,
                    batch_size=settings2["batch_size"],
                    shuffle=settings2["shuffle"],
                ))

                sampled_data = next(iter(train_loaders[0]))
                # assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.min() == 0
                assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.max() == 1

                val_loaders.append( LinkNeighborLoader(
                    data=val_data,
                    num_neighbors=settings2["num_neighbors"],
                    edge_label_index=(tuple(settings1["edge_types"]), val_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
                    edge_label=val_data[edge_types[0],edge_types[1],edge_types[2]].edge_label,
                    batch_size=settings2["batch_size"]*3,
                    shuffle=False,
                ))

                sampled_data = next(iter(val_loaders[0]))
                # assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.min() == 0
                assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.max() == 1

                test_loaders.append( LinkNeighborLoader(
                    data=test_data,
                    num_neighbors=settings2["num_neighbors"],
                    edge_label_index=(tuple(settings1["edge_types"]), test_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index),
                    edge_label=test_data[edge_types[0],edge_types[1],edge_types[2]].edge_label,
                    batch_size=settings2["batch_size"]*3,
                    shuffle=False,
                ))

                sampled_data = next(iter(test_loaders[0]))
                # assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.min() == 0
                assert sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label.max() == 1

            loaders = {"train" : train_loaders, "val": val_loaders, "test": test_loaders}

        elif stage == "inference":
            hdata = from_networkxwrapper_2_heterodata(nxdata_list)
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
        print(f"GNNWrapper: Defining GCN")
        settings = self.settings["gnn"]
        class GCN(torch.nn.Module):
            def __init__(self, hidden_channels):
                super().__init__()
                torch.manual_seed(1234)
                self.conv1 = SAGEConv(hidden_channels, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, hidden_channels)
                self.conv3 = SAGEConv(hidden_channels, hidden_channels)

            def forward(self, x, edge_index):
                h = self.conv1(x, edge_index).relu()
                h = self.conv2(h, edge_index)

                return h
            
        class Classifier(torch.nn.Module):
            def forward(self, x_input: Tensor, x_output: Tensor, edge_label_index: Tensor) -> Tensor:
                # Convert node embeddings to edge-level representations:
                edge_feat_input = x_input[edge_label_index[0]]
                edge_feat_output = x_output[edge_label_index[1]]

                # Apply dot-product to get a prediction per supervision edge:
                return (edge_feat_input * edge_feat_output).sum(dim=-1)

        class Model(torch.nn.Module):
            def __init__(self, hdata_loaders):
                super().__init__()
                # Since the dataset does not come with rich features, we also learn two
                # embedding matrices for users and movies:
                
                sampled_data = next(iter(hdata_loaders["train"][0]))
                
                # Instantiate homogeneous GNN:
                self.gnn = GCN(settings["hidden_channels"])

                # Convert GNN model into a heterogeneous variant:
                self.gnn = to_hetero(self.gnn, metadata=sampled_data.metadata())

                self.classifier = Classifier()

            def reset_embeddings(self, num_nodes, num_features, hidden_channels):
                self.ws_lin = torch.nn.Linear(num_features, hidden_channels)
                self.ws_emb = torch.nn.Embedding(num_nodes, hidden_channels)

            def forward(self, data: HeteroData) -> Tensor:
                x_dict = {
                "ws": self.ws_lin(data["ws"].x) + self.ws_emb(data["ws"].node_id),
                } 

                # `x_dict` holds feature matrices of all node types
                # `edge_index_dict` holds all edge indices of all edge types
                x_dict = self.gnn(x_dict, data.edge_index_dict)

                pred = self.classifier(
                    x_dict["ws"],
                    x_dict["ws"],
                    data["ws", "ws_same_room", "ws"].edge_label_index,
                )
                return pred
            
        self.model = Model(self.hdata_loaders)


    def train(self, verbose = False):
        print(f"GNNWrapper: Training")
        settings = self.settings["gnn"]
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        edge_types = tuple(self.settings["random_link_split"]["edge_types"])
        self.validate()
        if verbose:
            plt.show()

        for epoch in range(1, settings["epochs"]):
            total_loss = total_examples = 0

            for hdata_train_graph_loader in tqdm.tqdm(self.hdata_loaders["train"], colour="red"):
                self.model.reset_embeddings(hdata_train_graph_loader.data["ws"].num_nodes , settings["num_features"], settings["hidden_channels"])
                self.model = self.model.to(self.device)

                for sampled_data in hdata_train_graph_loader:
                    self.optimizer.zero_grad()
                    sampled_data.to(self.device)
                    pred = self.model(sampled_data)
                    loss = F.binary_cross_entropy_with_logits(pred, sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += float(loss) * pred.numel()
                    total_examples += pred.numel()

            print(f"Epoch: {epoch:03d}, Training Loss: {total_loss / total_examples:.4f}")
            if verbose:
                self.plot_values["train"]["loss"].append(total_loss / total_examples)
                plt.figure("Train loss")
                # plt.ylim([0, 1])
                plt.plot(np.array(self.plot_values["train"]["loss"]), 'r', label = "Loss")
                plt.xlabel('Epochs')
                plt.ylabel('%')
                plt.draw()
                plt.pause(0.001)
            self.validate(verbose)
        self.test()


    def validate(self,verbose = False):
        preds = []
        ground_truths = []
        mp_index_tuples = []
        settings = self.settings["gnn"]
        edge_types = tuple(self.settings["random_link_split"]["edge_types"])
        for hdata_val_graph_loader in self.hdata_loaders["val"]:
            self.model.reset_embeddings(hdata_val_graph_loader.data["ws"].num_nodes , settings["num_features"], settings["hidden_channels"])
            self.model = self.model.to(self.device)
            for sampled_data in hdata_val_graph_loader:
                with torch.no_grad():
                    sampled_data.to(self.device)
                    pred = self.model(sampled_data)
                    preds.append(pred)
                    ground_truths.append(sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label)
                    edge_label_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index
                    mp_index_tuples = mp_index_tuples + [(indexes[0], indexes[1]) for indexes in zip(edge_label_index[0], edge_label_index[1])]

        pred = torch.cat(preds, dim=0).cpu().numpy()
        pred_label = [1 if n>0.5 else 0 for n in pred]
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

        mp_edges = []
        for i,pair in enumerate(mp_index_tuples):
            mp_edges.append((pair[0], pair[1], {"type" : edge_types[1], "label": pred_label[i]}))

        auc = roc_auc_score(ground_truth, pred)
        print(f"Validation AUC: {auc:.4f}")
        accuracy, precission, recall, f1 = self.compute_metrics_from_all_predictions(pred_label, ground_truth, verbose= False)

        self.plot_values["val"]["auc"].append(auc)
        self.plot_values["val"]["acc"].append(accuracy)
        self.plot_values["val"]["prec"].append(precission)
        self.plot_values["val"]["rec"].append(recall)
        self.plot_values["val"]["f1"].append(f1)

        if verbose:
            plt.figure("Val Metrics")
            plt.clf()
            plt.ylim([0, 1])
            plt.plot(np.array(self.plot_values["val"]["acc"]), label = "Accuracy")
            plt.plot(np.array(self.plot_values["val"]["prec"]), label = "Precission")
            plt.plot(np.array(self.plot_values["val"]["rec"]), label = "Recall")
            plt.plot(np.array(self.plot_values["val"]["f1"]), label = "F1")
            plt.plot(np.array(self.plot_values["val"]["auc"]), label = "AUC")
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('%')
            plt.draw()
            plt.pause(0.001)

    def test(self):
        preds = []
        ground_truths = []
        settings = self.settings["gnn"]
        edge_types = tuple(self.settings["random_link_split"]["edge_types"])
        for hdata_test_graph_loader in self.hdata_loaders["test"]:
            self.model.reset_embeddings(hdata_test_graph_loader.data["ws"].num_nodes , settings["num_features"], settings["hidden_channels"])
            self.model = self.model.to(self.device)
            for sampled_data in hdata_test_graph_loader:
                with torch.no_grad():
                    sampled_data.to(self.device)
                    pred = self.model(sampled_data)
                    preds.append(pred)
                    ground_truths.append(sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label)

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)
        print(f"Test AUC: {auc:.4f}")

    
    def infer(self, nx_data, ground_truth = None):
        print("GNNWrapper: Inferring")
        device = "cpu"
        gnn_settings = self.settings["gnn"]
        rls_settings = self.settings["random_link_split"]

        # loader = self.preprocess_data([nx_data], "train")["val"][0]### TODO no loeader 
        sampled_data = self.preprocess_data(nx_data, "inference")
        edge_types = tuple(rls_settings["edge_types"])

        self.model = self.model.to(device)
        # self.model.reset_embeddings(loader.data["ws"].num_nodes , gnn_settings["num_features"], gnn_settings["hidden_channels"]) ### TODO no loeader 
        self.model.reset_embeddings(sampled_data["ws"].num_nodes , gnn_settings["num_features"], gnn_settings["hidden_channels"])
        pred_label = []
        edge_label_index_tuples = []
        edge_index_tuples = []
        
        # for sampled_data in tqdm.tqdm(loader, colour="green"): ### TODO No loader, also in creator function
        with torch.no_grad():  
            
            sampled_data.to(device)
            edge_label_index = sampled_data[edge_types[0],edge_types[1],edge_types[2]].edge_label_index.cpu().numpy()
            edge_label_index_tuples = edge_label_index_tuples + [(indexes[0], indexes[1]) for indexes in zip(edge_label_index[0], edge_label_index[1])]

            pred = self.model(sampled_data)
            pred_label = pred_label + [ 1 if n>0.5 else 0 for n in pred]

            # edge_index_tuples = edge_index_tuples + [(indexes[0], indexes[1]) for indexes in zip(edge_label_index[0], edge_label_index[1])] ### TODO Eliminate
        
        if isinstance(ground_truth, list):
            ground_truth_label = [ 1 if pair in ground_truth else 0 for pair in edge_label_index_tuples]
            accuracy, precission, recall, f1 = self.compute_metrics_from_all_predictions(pred_label, ground_truth_label, verbose= True)

        positive_edges_index = []
        for i,pair in enumerate(edge_label_index_tuples):
            positive_edges_index.append((pair[0], pair[1], {"type" : edge_types[1], "label": pred_label[i]}))

        return positive_edges_index

    def compute_metrics_from_all_predictions(self, pred_label, ground_truth_label, verbose = False):

        assert len(pred_label) == len(ground_truth_label)
    
        len_all_indexes = len(pred_label)
        len_predicted_positives = sum(pred_label)
        len_predicted_negatives = len_all_indexes - len_predicted_positives
        len_actual_positives = sum(ground_truth_label)
        len_actual_negatives = len_all_indexes - len_actual_positives

        true_positives = sum(compress(np.array(pred_label) == np.array(ground_truth_label), [True if n==1. else False for n in pred_label]))
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
            print(f"Accuracy {accuracy:.3f}, Precission {precission:.3f}, Recall {recall:.3f}, F1 {f1:.3f}")
        
        return accuracy, precission, recall, f1
    

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
