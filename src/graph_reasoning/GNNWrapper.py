import numpy as np
import torch, os, sys
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
from itertools import combinations
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import asyn_lpa_communities
import matplotlib.pyplot as plt
import torch.nn.init as init
import gc

from torchmetrics.classification import (MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAUROC, CalibrationError)

# from .FactorNN import FactorNN

# graph_reasoning_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_reasoning")
# sys.path.append(graph_reasoning_dir)
# from graph_reasoning.from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata
# graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
# sys.path.append(graph_datasets_dir)
# graph_matching_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_matching")
# sys.path.append(graph_matching_dir)

# graph_factor_nn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_factor_nn")
# sys.path.append(graph_factor_nn_dir)
from graph_reasoning.from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata
from graph_reasoning.MetricsSubplot import MetricsSubplot
from graph_reasoning.GNNs.v1.G_GNNv1 import G_GNNv1
from graph_reasoning.GNNs.v2.G_GNNv2 import G_GNNv2
from graph_reasoning.GNNs.v3.G_GNNv3 import G_GNNv3
from graph_datasets.graph_visualizer import visualize_nxgraph


class GNNWrapper():
    def __init__(self, settings, report_path, logger = None, ID=0) -> None:
        print(f"GNNWrapper{ID}: ", Fore.BLUE + "Initializing" + Fore.WHITE)
        self.settings = settings
        self.target_concept = settings["report"]["target_concept"]
        self.report_path = report_path
        self.logger = logger
        self.ID = ID
        self.epochs_verbose_rate = 5
        self.pth_path = os.path.join(self.report_path,'model.pth')
        self.best_pth_path = os.path.join(self.report_path,'model_best.pth')
        self.set_cuda_device()

        if logger:
            self.logger.info(f"GNNWrapper{self.ID} : torch device => {self.device}")
        else:
            print(f"GNNWrapper{self.ID}: ", Fore.BLUE + f"Torch device => {self.device}" + Fore.WHITE)
        metric_values_dict = {"loss_avg" : torch.Tensor([]).to(self.device), "auc" : torch.Tensor([]).to(self.device), "acc" : torch.Tensor([]).to(self.device), "prec" : torch.Tensor([]).to(self.device), "rec" : torch.Tensor([]).to(self.device), "f1" : torch.Tensor([]).to(self.device), "pred_pos_rate" : torch.Tensor([]).to(self.device), "gt_pos_rate": torch.Tensor([]).to(self.device), "score":torch.Tensor([]).to(self.device), "e_uncertainty": torch.Tensor([]).to(self.device),"a_uncertainty": torch.Tensor([]).to(self.device),"ece": torch.Tensor([]).to(self.device)}
        self.metric_values = {"train" : copy.deepcopy(metric_values_dict), "val" : copy.deepcopy(metric_values_dict),\
                            "test" : copy.deepcopy(metric_values_dict), "inference" : copy.deepcopy(metric_values_dict),}

        metric_names_map = {"train RoomWall Confusion Metrics": 0, "train RoomWall Learning Metrics": 1,\
                          "val RoomWall Confusion Metrics": 2, "val RoomWall Learning Metrics": 3,\
                          "test RoomWall Confusion Metrics": 4, "test RoomWall Learning Metrics": 5}
        self.metrics_subplot = MetricsSubplot(name=str(ID), nrows=3, ncols=2, plot_names_map=metric_names_map)

        graph_names_map = {"train RoomWall inference": 0, "train RoomWall Uncertainty inference": 1, "train Inference rooms graph": 2,"train Inference walls graph": 3,\
                          "val RoomWall inference": 4, "val RoomWall Uncertainty inference": 5, "val Inference rooms graph": 6,"val Inference walls graph": 7,\
                          "test RoomWall inference": 8, "test RoomWall Uncertainty inference": 9,"test Inference rooms graph": 10,"test Inference walls graph":11}
        self.graphs_subplot = MetricsSubplot(name=str(ID), nrows=3, ncols=4, plot_names_map=graph_names_map)


    def set_cuda_device(self):
        min_free_memory_mb = 1024
        if self.settings["gnn"]["use_cuda"]:
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                available_devices = []

                # Check each GPU's memory
                for device_id in range(num_devices):
                    free_memory, total_memory = torch.cuda.mem_get_info(device_id) 
                    free_memory_mb = free_memory / (1024 * 1024)
                    total_memory_mb = total_memory / (1024 * 1024)  ## HPC devices total memory: 15.7657470703125 GB
                    usage = free_memory_mb/total_memory_mb
                    # print(f"Device {device_id}: {free_memory_mb:.2f} MB free out of {total_memory_mb:.2f} MB total, a {usage}% usage")

                    if free_memory_mb >= min_free_memory_mb:
                        available_devices.append((device_id, free_memory_mb))

                if available_devices:
                    best_device = max(available_devices, key=lambda x: x[1])
                    # print(f"Selected Device {best_device[0]} with {best_device[1]:.2f} MB free memory.")
                    self.device = torch.device(f'cuda:{best_device[0]}')

                # print("No GPU has sufficient memory. Falling back to CPU.")
            else:
                print("No CUDA devices available. Using CPU.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
    def set_nxdataset(self, nxdataset, hdataset):
        self.nxdataset = nxdataset
        if hdataset == None:
            print(f"GNNWrapper{self.ID}: ", Fore.BLUE + f"Prepocessing nxdataset" + Fore.WHITE)
            self.hdataset = self.preprocess_nxdataset(nxdataset)
        else:
            self.hdataset = hdataset

        for hdataset in self.hdataset.values():
            for hdata in hdataset:
                hdata.to(self.device)

    def preprocess_nxdataset(self, nxdataset):
        hdataset = {}
        for tag in nxdataset.keys():
            hdataset_key = []
            for nx_data in (pbar := tqdm.tqdm(nxdataset[tag], colour="blue")):
                hdata = from_networkxwrapper_2_heterodata(nx_data)
                hdataset_key.append(hdata)

            hdataset[tag] = hdataset_key

        return hdataset
    
    def visualize_hetero_features(self):
        """
        Visualizes the feature distributions for a list of HeteroData objects.
        Each node type and edge type gets its own row in the plot grid.
        Each feature gets its own subplot within the respective row.
        """
        hdata_list = self.hdataset["train"]
        # First, gather all unique node and edge types across the dataset
        node_types = set()
        edge_types = set()
        for hdata in hdata_list:
            node_types.update(hdata.node_types)
            edge_types.update(hdata.edge_types)
        
        node_types = sorted(node_types)
        edge_types = sorted(edge_types)
        
        num_node_features = max(hdata[node_types[0]].x.size(1) for hdata in hdata_list)
        num_edge_features = max(hdata[edge_types[0]].x.size(1) for hdata in hdata_list)

        # Calculate the total rows for subplots
        total_rows = len(node_types) + len(edge_types)
        total_features = max(num_node_features, num_edge_features)
        
        # Initialize the plot
        fig, axes = plt.subplots(total_rows, total_features, figsize=(total_features * 4, total_rows * 3))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
        row_idx = 0
        
        # Plot node features for each node type
        for node_type in node_types:
            for feature_idx in range(num_node_features):
                ax = axes[row_idx, feature_idx]
                for hdata in hdata_list:
                    node_features = hdata[node_type].x[:, feature_idx].cpu().numpy()
                    ax.hist(node_features, bins=30, alpha=0.5, label=f"Graph {hdata_list.index(hdata)}")
                ax.set_title(f"{node_type} - Feature {feature_idx}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
            row_idx += 1
        
        # Plot edge features for each edge type
        for edge_type in edge_types:
            for feature_idx in range(num_edge_features):
                ax = axes[row_idx, feature_idx]
                for hdata in hdata_list:
                    edge_features = hdata[edge_type].x[:, feature_idx].cpu().numpy()
                    ax.hist(edge_features, bins=30, alpha=0.5, label=f"Graph {hdata_list.index(hdata)}")
                ax.set_title(f"{edge_type} - Feature {feature_idx}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
            row_idx += 1

        plt.savefig(os.path.join(self.report_path,f'gnn_input_features.png'), bbox_inches='tight')

            
    def define_GCN(self):
        print(f"GNNWrapper{self.ID}: ", Fore.BLUE + "Defining GCN" + Fore.WHITE)
        self.model = G_GNNv2(self.settings, self.logger)
        self.model.to(self.device)
        # self.best_model = copy.deepcopy(self.model)
        self.model.set_use_MC_dropout(False)

    def train(self, verbose = False):
        print(f"GNNWrapper{self.ID}: ", Fore.BLUE + "Training" + Fore.WHITE)

        best_val_loss = float('inf')
        training_settings = self.settings["training"]
        patience = training_settings["patience"]
        trigger_times = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings["gnn"]["lr"])
        self.criterion = torch.nn.CrossEntropyLoss()
        original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        # num_classes = self.settings["gnn"]["decoder"]["classifier"]["classes"]
        num_classes = self.settings["gnn"]["decoder"]["output_channels"]
        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]

        accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(self.device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(self.device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(self.device)
        auroc_metric = MulticlassAUROC(num_classes=num_classes, average='macro').to(self.device)
        calibration_metric = CalibrationError(task="multiclass", num_classes=num_classes, n_bins=15, norm="l1").to(self.device)

        for epoch in (pbar := tqdm.tqdm(range(1, training_settings["epochs"]), colour="blue")):
            self.epoch = epoch
            accuracy_metric.reset()
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()
            auroc_metric.reset()
            calibration_metric.reset()
            a_uncertainty_metric = torch.empty(0, device=self.device)
            # e_uncertainty_metric = torch.empty(0, device=self.device)

            pbar.set_description(f"Epoch{self.ID}")
            total_loss = total_examples = 0
            # start_time = time.time()
            for i, hdata in enumerate(self.hdataset["train"]):
                self.optimizer.zero_grad()
                # sub_start_time = time.time()
                # sub_part_1_end = time.time()
                logits, log_var = self.model(hdata.x_dict, hdata.edge_index_dict,hdata.edge_label_dict)
                # sub_part_2_end = time.time()
                gt = hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label[hdata.edge_label_dict["edge_index_to_edge_label_index"]]
                # loss = self.combined_classification_loss(logits, log_var, gt)
                loss = self.base_classification_loss(logits, gt)
                # sub_part_3_end = time.time()
                loss.backward()
                # sub_part_4_end = time.time()
                self.optimizer.step()
                # sub_part_5_end = time.time()
                total_loss += loss * logits.numel()
                total_examples += logits.numel()

                with torch.no_grad():
                    probs = F.softmax(logits, dim=1) 

                    # print(f'dbg log_var {log_var}')
                    # min_max_var_range = max(var) - min(var)
                    # a_uncertainty = (var - min(var)) / min_max_var_range
                    a_uncertainty = log_var
                    # print(f"dbg probs {probs}")
                    # print(f"dbg var {var}")

                    accuracy_metric.update(probs, gt)
                    precision_metric.update(probs, gt)
                    recall_metric.update(probs, gt)
                    f1_metric.update(probs, gt)
                    auroc_metric.update(probs, gt)
                    calibration_metric.update(probs, gt)
                    a_uncertainty_metric = torch.cat((a_uncertainty_metric, a_uncertainty))
                    # entropy, variance = self.compute_output_entropy(hdata.x_dict, hdata.edge_index_dict, hdata.edge_label_dict,num_samples=5)
                    # avg_e_entropy = entropy.mean().view(1)
                    # e_uncertainty_metric = torch.cat((e_uncertainty_metric, avg_e_entropy))

                sub_part_6_end = time.time()
                if verbose and epoch % self.epochs_verbose_rate == 0 and i == len(self.hdataset["train"]) - 1 :
                    preds = np.argmax(probs.detach().cpu().numpy(), axis=1)
                    color_code = ["black", "blue", "orange"]
                    predicted_edges_last_graph = [(ei[0], ei[1], {"type" : original_edge_types[preds[i]],\
                                                "label": preds[i], "viz_feat": color_code[preds[i]], "linewidth":0.5 if preds[i]==0 else 1.5,\
                                                "alpha":0.3 if preds[i]==0 else 1.}) for i, ei in enumerate(hdata.edge_label_dict["edge_label_index_tuples_compressed"])]
                # print(f"sub part 1 took {sub_part_1_end - sub_start_time:.4f} seconds.")
                # print(f"sub part 2 took {sub_part_2_end - sub_part_1_end:.4f} seconds.")
                # print(f"sub part 3 took {sub_part_3_end - sub_part_2_end:.4f} seconds.")
                # print(f"sub part 4 took {sub_part_4_end - sub_part_3_end:.4f} seconds.")
                # print(f"sub part 5 took {sub_part_5_end - sub_part_4_end:.4f} seconds.")
                # print(f"sub part 6 took {sub_part_6_end - sub_part_5_end:.4f} seconds.")

            # part_1_end = time.time()

            # print(f"dbg len(probs_in_train_dataset) {probs_in_train_dataset[:50]}")
            # accuracy, precission, recall, f1, auc = self.compute_metrics_from_all_predictions(gt_in_train_dataset, probs_in_train_dataset, verbose= False)\
            # loss_avg = total_loss / total_examples
            loss_avg = total_loss / max(1, total_examples)
            train_accuracy = accuracy_metric.compute()
            train_precision = precision_metric.compute()
            train_recall = recall_metric.compute()
            train_f1 = f1_metric.compute()
            train_auc = auroc_metric.compute()
            train_calibration = calibration_metric.compute()
            train_a_uncertainty = a_uncertainty_metric.mean()
            # train_mc_entropy = e_uncertainty_metric.mean()
            w1, w2, w3, w4 = 0.35, 0.25, 0.15, 0.25
            w1, w2, w3, w4 = 0.34, 0.33, 0.33, 0.0
            composite_score = (w1 * loss_avg +
                            w2 * (1 - train_auc) +
                            w3 * train_calibration +
                            w4 * train_a_uncertainty)

            # Log metricss
            self.metric_values["train"]["auc"] = torch.cat((self.metric_values["train"]["auc"], train_auc.unsqueeze(0)))
            self.metric_values["train"]["acc"] = torch.cat((self.metric_values["train"]["acc"], train_accuracy.unsqueeze(0)))
            self.metric_values["train"]["prec"] = torch.cat((self.metric_values["train"]["prec"], train_precision.unsqueeze(0)))
            self.metric_values["train"]["rec"] = torch.cat((self.metric_values["train"]["rec"], train_recall.unsqueeze(0)))
            self.metric_values["train"]["f1"] = torch.cat((self.metric_values["train"]["f1"], train_f1.unsqueeze(0)))
            self.metric_values["train"]["loss_avg"] = torch.cat((self.metric_values["train"]["loss_avg"], loss_avg.unsqueeze(0)))
            # self.metric_values["train"]["e_uncertainty"] = torch.cat((self.metric_values["train"]["e_uncertainty"], train_mc_entropy.unsqueeze(0)))
            self.metric_values["train"]["a_uncertainty"] = torch.cat((self.metric_values["train"]["a_uncertainty"], train_a_uncertainty.unsqueeze(0)))
            self.metric_values["train"]["ece"] = torch.cat((self.metric_values["train"]["ece"], train_calibration.unsqueeze(0)))
            self.metric_values["train"]["score"] = torch.cat((self.metric_values["train"]["score"], composite_score.unsqueeze(0)))

            if verbose and epoch % self.epochs_verbose_rate == 0:
                ### Metrics
                self.plot_metrics("train")
                # if self.settings["report"]["save"]:
                #     plt.savefig(os.path.join(self.report_path,f'train_metrics.png'), bbox_inches='tight')

                ### inference - Inference
                merged_graph = self.merge_predicted_edges(copy.deepcopy(self.nxdataset["train"][-1]), predicted_edges_last_graph)
                fig = visualize_nxgraph(merged_graph, image_name = f"train {self.target_concept} inference", include_node_ids= False)
                self.graphs_subplot.update_plot_with_figure(f"train {self.target_concept} inference", fig, square_it = True)

                if self.target_concept == "RoomWall":
                    clusters, inferred_graph = self.cluster_RoomWall(merged_graph, "train")

            val_loss = self.validate("val", verbose and epoch % self.epochs_verbose_rate == 0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                self.best_model = copy.deepcopy(self.model)
                self.save_best_model()
            else:
                trigger_times += 1
            if trigger_times >= patience:
                print(f"GNNWrapper{self.ID}: ", Fore.BLUE + "Early stopping triggered" + Fore.WHITE)
                self.plot_metrics("train")
                # test_loss = self.validate( "test", verbose= True)
                break

            # self.save_best_model()

        self.plot_metrics("train")
        self.model = self.best_model
        self.validate("val", True)
        self.metrics_subplot.close()
        self.graphs_subplot.close()
        test_score = self.validate( "test", verbose= True)
        return test_score


    def validate(self,tag,verbose = False):
        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]
        original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        num_classes = len(original_edge_types)

        accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(self.device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(self.device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(self.device)
        auroc_metric = MulticlassAUROC(num_classes=num_classes).to(self.device)
        # calibration_metric = CalibrationError(task="multiclass", num_classes=self.settings["gnn"]["decoder"]["classifier"]["classes"],n_bins=15, norm="l1", ).to(self.device)
        calibration_metric = CalibrationError(task="multiclass", num_classes=self.settings["gnn"]["decoder"]["output_channels"],n_bins=15, norm="l1", ).to(self.device)
        a_uncertainty_metric = torch.empty(0, device=self.device)
        e_uncertainty_metric = torch.empty(0, device=self.device)

        total_loss = total_examples = 0

        for i, hdata in enumerate(self.hdataset[tag]):
            total_loss = total_examples = 0

            with torch.no_grad():

                logits, log_var = self.model(hdata.x_dict, hdata.edge_index_dict, hdata.edge_label_dict)
                gt = hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label[hdata.edge_label_dict["edge_index_to_edge_label_index"]]
                # loss = self.combined_classification_loss(logits, log_var, gt)
                loss = self.base_classification_loss(logits, gt)
                total_loss += loss * logits.numel()
                total_examples += logits.numel()

                probs = F.softmax(logits, dim=1)
                # log_var = torch.clamp(log_var, min=-10.0, max=10.0)  # Clamping log_var
                # var = torch.exp(-log_var).squeeze()
                # log_var_scaled = torch.log(var + 1)
                # min_max_var_range = max(var) - min(var)
                # a_uncertainty = (var - min(var)) / min_max_var_range
                a_uncertainty = log_var

                accuracy_metric.update(probs, gt)
                precision_metric.update(probs, gt)
                recall_metric.update(probs, gt)
                f1_metric.update(probs, gt)
                auroc_metric.update(probs, gt)
                calibration_metric.update(probs, gt)
                a_uncertainty_metric = torch.cat((a_uncertainty_metric, a_uncertainty))

                if tag == 'test' or self.epoch % 20 == 1:
                    mc_entropy, variance = self.compute_output_entropy(hdata.x_dict, hdata.edge_index_dict, hdata.edge_label_dict,num_samples=5)#.mean().view(1)
                    # min_max_var_range = max(mc_entropy) - min(mc_entropy)
                    # e_uncertainty = (mc_entropy - min(mc_entropy)) / min_max_var_range
                    avg_e_uncertainty = mc_entropy.mean().view(1)

                    # # DBG
                    # fig, ax = plt.subplots()
                    # ax.hist(mc_entropy.cpu(), bins=30, edgecolor='black')
                    # fig.savefig(os.path.join(self.report_path,f'mc_entropy_histogram.png'), bbox_inches='tight')
                    # fig, ax = plt.subplots()
                    # ax.hist(var.cpu(), bins=30, edgecolor='black')
                    # fig.savefig(os.path.join(self.report_path,f'var_histogram.png'), bbox_inches='tight')
                    # # END DBG
                else: 
                    avg_e_uncertainty = self.metric_values[tag]["e_uncertainty"][-1].view(1)

                e_uncertainty_metric = torch.cat((e_uncertainty_metric, avg_e_uncertainty))


            if verbose and i == len(self.hdataset[tag]) - 1:
                preds = np.argmax(probs.detach().cpu().numpy(), axis=1)
                color_code = ["black", "blue", "orange"]
                predicted_edges_last_graph = [(ei[0], ei[1], {"type" : original_edge_types[preds[i]],\
                                            "label": preds[i], "viz_feat": color_code[preds[i]], "linewidth":0.5 if preds[i]==0 else 1.5,\
                                            "alpha":0.3 if preds[i]==0 else 1.}) for i, ei in enumerate(hdata.edge_label_dict["edge_label_index_tuples_compressed"])]
            
                mc_entropy, variance = self.compute_output_entropy(hdata.x_dict, hdata.edge_index_dict, hdata.edge_label_dict,num_samples=10)
                # min_max_var_range = max(mc_entropy) - min(mc_entropy)
                # e_uncertainty = (mc_entropy - min(mc_entropy)) / min_max_var_range
                e_uncertainty = mc_entropy.mean().view(1)
                # common_uncertainty_metric = np.array(copy.deepcopy((e_uncertainty + a_uncertainty) / 2).cpu())
                e_certainty_metric = np.clip(np.ones(mc_entropy.size()) - np.array(copy.deepcopy((mc_entropy).cpu())), 0, 1)
                
                predicted_edges_uncertainty_last_graph = [(ei[0], ei[1], {"type" : original_edge_types[preds[i]],\
                                            "label": preds[i], "viz_feat": color_code[preds[i]], "linewidth":e_certainty_metric[i]*1.5,\
                                            "alpha":e_certainty_metric[i]}) for i, ei in enumerate(hdata.edge_label_dict["edge_label_index_tuples_compressed"])]
            
        # accuracy, precission, recall, f1, auc = self.compute_metrics_from_all_predictions(gt_in_val_dataset, probs_in_val_dataset, verbose= False)
        # loss_avg = total_loss / total_examples
        loss_avg = total_loss / max(1, total_examples)
        val_accuracy = accuracy_metric.compute()
        val_precision = precision_metric.compute()
        val_recall = recall_metric.compute()
        val_f1 = f1_metric.compute()
        val_auc = auroc_metric.compute()
        val_calibration = calibration_metric.compute()
        val_a_uncertainty = a_uncertainty_metric.mean()
        val_e_uncertainty = e_uncertainty_metric.mean()

        w1, w2, w3, w4, w5 = 0.35, 0.25, 0.15, 0.125, 0.125
        w1, w2, w3, w4, w5 = 0.25, 0.25, 0.25, 0.0, 0.25
        score = (w1 * loss_avg +
                w2 * (1 - val_auc) +
                w3 * val_calibration +
                w4 * val_a_uncertainty + 
                w5 * val_e_uncertainty)


        self.metric_values[tag]["auc"] = torch.cat((self.metric_values[tag]["auc"], val_auc.unsqueeze(0)))
        self.metric_values[tag]["acc"] = torch.cat((self.metric_values[tag]["acc"], val_accuracy.unsqueeze(0)))
        self.metric_values[tag]["prec"] = torch.cat((self.metric_values[tag]["prec"], val_precision.unsqueeze(0)))
        self.metric_values[tag]["rec"] = torch.cat((self.metric_values[tag]["rec"], val_recall.unsqueeze(0)))
        self.metric_values[tag]["f1"] = torch.cat((self.metric_values[tag]["f1"], val_f1.unsqueeze(0)))
        self.metric_values[tag]["loss_avg"] = torch.cat((self.metric_values[tag]["loss_avg"], loss_avg.unsqueeze(0)))
        self.metric_values[tag]["e_uncertainty"] = torch.cat((self.metric_values[tag]["e_uncertainty"], val_e_uncertainty.unsqueeze(0)))
        self.metric_values[tag]["a_uncertainty"] = torch.cat((self.metric_values[tag]["a_uncertainty"], val_a_uncertainty.unsqueeze(0)))
        self.metric_values[tag]["ece"] = torch.cat((self.metric_values[tag]["ece"], val_calibration.unsqueeze(0)))
        self.metric_values[tag]["score"] = torch.cat((self.metric_values[tag]["score"], score.unsqueeze(0)))

        if tag == "test":
            self.metric_values[tag]["auc"] = torch.cat((self.metric_values[tag]["auc"], val_auc.unsqueeze(0)))
            self.metric_values[tag]["acc"] = torch.cat((self.metric_values[tag]["acc"], val_accuracy.unsqueeze(0)))
            self.metric_values[tag]["prec"] = torch.cat((self.metric_values[tag]["prec"], val_precision.unsqueeze(0)))
            self.metric_values[tag]["rec"] = torch.cat((self.metric_values[tag]["rec"], val_recall.unsqueeze(0)))
            self.metric_values[tag]["f1"] = torch.cat((self.metric_values[tag]["f1"], val_f1.unsqueeze(0)))
            self.metric_values[tag]["loss_avg"] = torch.cat((self.metric_values[tag]["loss_avg"], loss_avg.unsqueeze(0)))
            self.metric_values[tag]["e_uncertainty"] = torch.cat((self.metric_values[tag]["e_uncertainty"], val_e_uncertainty.unsqueeze(0)))
            self.metric_values[tag]["a_uncertainty"] = torch.cat((self.metric_values[tag]["a_uncertainty"], val_a_uncertainty.unsqueeze(0)))
            self.metric_values[tag]["ece"] = torch.cat((self.metric_values[tag]["ece"], val_calibration.unsqueeze(0)))
            self.metric_values[tag]["score"] = torch.cat((self.metric_values[tag]["score"], score.unsqueeze(0)))

        if verbose:
            ### Metrics
            self.plot_metrics(tag)

            ### inference - Inference
            merged_graph = self.merge_predicted_edges(copy.deepcopy(self.nxdataset[tag][-1]), predicted_edges_last_graph)
            fig = visualize_nxgraph(merged_graph, image_name = f"{tag} {self.target_concept} inference", include_node_ids= False)
            self.graphs_subplot.update_plot_with_figure(f"{tag} {self.target_concept} inference", fig, square_it = True)
            del fig

            # predicted_edges_uncertainty_last_graph
            uncertainty_graph = self.merge_predicted_edges(copy.deepcopy(self.nxdataset[tag][-1]), predicted_edges_uncertainty_last_graph)
            fig = visualize_nxgraph(uncertainty_graph, image_name = f"{tag} {self.target_concept} Uncertainty inference", include_node_ids= False)
            self.graphs_subplot.update_plot_with_figure(f"{tag} {self.target_concept} Uncertainty inference", fig, square_it = True)
            del fig

            if self.target_concept == "RoomWall":
                clusters, inferred_graph = self.cluster_RoomWall(merged_graph, tag)

            if self.settings["report"]["save"]:
                self.metrics_subplot.save(os.path.join(self.report_path,f'metric {self.target_concept} subplot.png'))
                self.graphs_subplot.save(os.path.join(self.report_path,f'graphs {self.target_concept} subplot.png'))
            
                # plt.savefig(os.path.join(self.report_path,f'{self.target_concept} subplot.png'), bbox_inches='tight')


        return score


    def infer(self, nx_data, verbose, use_gt = False, to_sgraph = False):

        self.model.eval()
        ncols = 3
        plot_names_map = {"infer RoomWall inference": 0, "infer Uncertainty inference": 1,"infer Inference rooms graph": 2,"infer Inference walls graph": 3}
        if to_sgraph:
            plot_names_map["to Sgraph"] = 3
            ncols = 4

        self.graphs_subplot = MetricsSubplot("infer", nrows=1, ncols=ncols, plot_names_map=plot_names_map)

        original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        color_code = ["black", "blue", "brown"]

        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]
        hdata = from_networkxwrapper_2_heterodata(nx_data)

        with torch.no_grad():
            hdata.to(self.device)
            node_key = list(hdata.edge_index_dict.keys())[0][0]
            edge_key = list(hdata.edge_index_dict.keys())[0][1]
            edge_index = copy.copy(hdata.edge_index_dict[node_key, edge_key, node_key]).cpu().numpy()
            edge_index_tuples = list(zip(edge_index[0], edge_index[1]))
            edge_label_index = copy.copy(hdata.edge_index_dict[node_key, edge_key, node_key]).cpu().numpy()
            edge_label_index_tuples_compressed = np.array(list({tuple(sorted((edge_label_index[0, i], edge_label_index[1, i]))) for i in range(edge_label_index.shape[1])}))
            edge_label_index_tuples_compressed_inversed = edge_label_index_tuples_compressed[:, ::-1]
            src, dst = edge_label_index_tuples_compressed[:,0], edge_label_index_tuples_compressed[:,1]
            edge_index_to_edge_label_index = [np.argwhere((edge_label_index_single == edge_index_tuples).all(1))[0][0] for edge_label_index_single in edge_label_index_tuples_compressed]
            edge_index_to_edge_label_index_inversed = [np.argwhere((edge_label_index_single == edge_index_tuples).all(1))[0][0] for edge_label_index_single in edge_label_index_tuples_compressed_inversed]
            edge_label_dict = {"src":src, "dst":dst, "edge_index_to_edge_label_index":edge_index_to_edge_label_index, "edge_index_to_edge_label_index_inversed":edge_index_to_edge_label_index_inversed}
                
            logits, log_var = self.model(hdata.x_dict, hdata.edge_index_dict, edge_label_dict)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            # self.logger.info(f'sbg log_var {log_var}')
            var = torch.exp(-log_var).squeeze()
            # self.logger.info(f'sbg var {var}')
            min_max_var_range = max(var) - min(var)
            # self.logger.info(f'sbg min_max_log_var_range {min_max_var_range}')
            uncertainty = (var + min(var)) / min_max_var_range
            # self.logger.info(f'sbg var + min(var) {var - min(var)}')
            # self.logger.info(f'sbg uncertainty {uncertainty}')

            edge_index = list(hdata[edge_types[0],edge_types[1],edge_types[2]].edge_index.cpu().numpy())
            edge_index = np.array(list(zip(edge_index[0], edge_index[1])))
            if use_gt:
                preds = hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label.cpu().numpy()

            predicted_edges_last_graph = [(ei[0], ei[1], {"type" : original_edge_types[preds[i]],\
                                        "label": preds[i], "viz_feat": color_code[preds[i]], "linewidth":0.5 if preds[i]==0 else 1.5,\
                                        "alpha":0.3 if preds[i]==0 else 1.}) for i, ei in enumerate(edge_label_index_tuples_compressed)]
            
            merged_graph = self.merge_predicted_edges(copy.deepcopy(nx_data), predicted_edges_last_graph)
            fig = visualize_nxgraph(merged_graph, image_name = f"infer {self.target_concept} inference")
            self.graphs_subplot.update_plot_with_figure(f"infer {self.target_concept} inference", fig, square_it = True)
            
            if self.target_concept == "RoomWall":
                clusters, inferred_graph = self.cluster_RoomWall(merged_graph, "infer")
                
            if self.settings["report"]["save"]:
                self.graphs_subplot.save(os.path.join(self.report_path,f'graphs {self.target_concept} subplot.png'))
            
        return clusters
    
    def compute_output_entropy(self, x_dict, edge_index_dict, edge_label_index_tuples_compressed, num_samples=25):

        def mc_forward(x_dict, edge_index_dict, edge_label_index_tuples_compressed, num_samples):
            """
            Run the model num_samples times with MC dropout enabled.
            
            Args:
                model: your GNN model.
                x_dict, edge_index_dict, edge_label_index_tuples_compressed: input data.
                num_samples: number of stochastic forward passes.
            
            Returns:
                A tensor of shape [num_samples, ...] containing the outputs from each run.
            """
            self.model.set_use_MC_dropout(True)
            self.model.eval()
            logits_list = []
            log_var_list = []
            with torch.no_grad():
                for _ in range(num_samples):
                    logits, log_var = self.model(x_dict, edge_index_dict, edge_label_index_tuples_compressed)
                    logits_list.append(logits)
                    log_var_list.append(log_var)
            self.model.set_use_MC_dropout(False)
            logits_stack = torch.stack(logits_list, dim=0)
            log_var_stack = torch.stack(log_var_list, dim=0)
            
            return logits_stack, log_var_stack
            
        # Run MC dropout inference:
        logits_stack, log_var_stack = mc_forward(x_dict, edge_index_dict, edge_label_index_tuples_compressed, num_samples=num_samples)
        # Apply softmax to convert logits to probabilities for each MC sample
        mc_probs = F.softmax(logits_stack, dim=-1)  # shape: [num_samples, num_edges, num_classes]

        # Compute the mean prediction over samples:
        mean_probs = mc_probs.mean(dim=0)  # shape: [num_edges, num_classes]

        # Compute the predictive entropy as a measure of uncertainty:
        epsilon = 1e-8  # for numerical stability
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=-1)  # shape: [num_edges]

        # Optionally, you could also compute the variance of the probabilities:
        variance = mc_probs.var(dim=0)  # shape: [num_edges, num_classes]
        return entropy, variance
    

    def combined_classification_loss(self, logits, log_var, targets):
        # Standard cross-entropy loss on the logits:
        ce_loss = self.criterion(logits, targets)  # shape: [num_samples]
        # Scale the loss by the predicted uncertainty:
        # We use exp(-log_var) as an attenuation factor
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        var = torch.exp(-log_var).squeeze()
        weighted_loss = ce_loss * var
        print(f"dbg ce_loss {ce_loss}")
        print(f"dbg weighted_loss {weighted_loss}")
        print(f"dbg log_var {log_var}")
        # Add a regularization term to prevent the network from trivially increasing variance
        reg_term = 0.5 * log_var.squeeze()
        # loss = torch.mean(weighted_loss + reg_term)
        loss = weighted_loss + reg_term
        print(f"dbg reg_term {reg_term}")
        print(f"dbg loss {loss}")
        return loss
    
    def base_classification_loss(self, logits, targets):
        loss = self.criterion(logits, targets)  # shape: [num_samples]
        return loss

    
    # def compute_ece(probs, targets, n_bins=10):
    #     # probs: [num_samples, num_classes] tensor of predicted probabilities
    #     # targets: [num_samples] tensor of true class indices
    #     confidences, predictions = probs.max(dim=1)
    #     accuracies = predictions.eq(targets)
    #     bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    #     ece = 0.0
    #     n = probs.size(0)
    #     for i in range(n_bins):
    #         # Find indices of predictions in bin i
    #         in_bin = confidences.gt(bin_boundaries[i]) * confidences.le(bin_boundaries[i + 1])
    #         prop_in_bin = in_bin.float().mean()
    #         if prop_in_bin.item() > 0:
    #             accuracy_in_bin = accuracies[in_bin].float().mean()
    #             avg_confidence_in_bin = confidences[in_bin].mean()
    #             ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    #     return ece


    def compute_metrics_from_all_predictions(self, ground_truth_label, prob_label, verbose = False):

        pred_label = np.argmax(prob_label, axis=1)
        assert len(pred_label) == len(ground_truth_label)

        accuracy = accuracy_score(ground_truth_label, pred_label)
        precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth_label, pred_label, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(ground_truth_label, pred_label)

        roc_auc = roc_auc_score(ground_truth_label, prob_label, multi_class='ovr')
        
        if verbose:
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}')
            print('Confusion Matrix:\n', conf_matrix)
            print(f'ROC AUC Score: {roc_auc}')
        
        return accuracy, precision, recall, f1_score, roc_auc#, gt_pos_rate, pred_pos_rate


    def merge_predicted_edges(self, unparented_base_graph, predictions):
        unparented_base_graph = copy.deepcopy(unparented_base_graph)
        unparented_base_graph.remove_all_edges()
        unparented_base_graph.add_edges(predictions)
        return unparented_base_graph
    

    def plot_metrics(self, tag):
        ["acc", "prec", "rec", "f1", "auc", "loss_avg", "a_uncertainty", "e_uncertainty", "ece", "score"]
        fig = plt.figure(f"{self.report_path} Metrics")
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_ylim([0, 1])
        label_mapping = {"acc": "Accuracy", "prec":"Precission", "rec":"Recall", "f1":"F1", "auc":"AUC", "ece":"ECE","e_uncertainty":"E-uncertainty"}
        color_mapping = {"acc": "orange", "prec":"green", "rec":"red", "f1":"purple", "auc":"brown", "ece":"yellow", "e_uncertainty":"cyan"}
        for metric_name in label_mapping.keys():
            ax.plot(np.array(self.metric_values[tag][metric_name].cpu().detach().numpy()), label = label_mapping[metric_name], color = color_mapping[metric_name])
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Rate')
        self.metrics_subplot.update_plot_with_figure(f"{tag} RoomWall Confusion Metrics", fig, square_it = False)
        plt.close(fig)

        fig = plt.figure(f"{self.report_path} Graphs")
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_ylim([0, 1])
        label_mapping = {"loss_avg":"Loss avg","a_uncertainty":"A-uncertainty", "score":"Score"}
        color_mapping = {"loss_avg":"blue","a_uncertainty":"green", "score":"black"}
        for metric_name in label_mapping.keys():
            ax.plot(np.array(self.metric_values[tag][metric_name].cpu().detach().numpy()), label = label_mapping[metric_name], color = color_mapping[metric_name])
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Rate')
        self.metrics_subplot.update_plot_with_figure(f"{tag} RoomWall Learning Metrics", fig, square_it = False)
        plt.close(fig)
        

    def cluster_rooms(self, old_graph):
        graph = copy.deepcopy(old_graph)
        graph = graph.filter_graph_by_edge_attributes({"type":"ws_same_room"})
        graph.to_undirected(type= "smooth")
        
        filtered_graph = copy.deepcopy(graph)
        # filtered_graph.filter_graph_by_degree_range([2, graph.get_degree_range()[1]])
        # visualize_nxgraph(graph, image_name = "room clustering", visualize_alone= True)

        def cluster_by_cycles(full_graph):
            all_cycles = []
            max_cycle_length = 10
            min_cycle_length = 2
            def iterative_cluster_rooms(working_graph, desired_cycle_length):
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

            for desired_cycle_length in reversed(range(min_cycle_length, max_cycle_length+1)):
                start_time = time.time()
                _, selected_cycles = iterative_cluster_rooms(full_graph, desired_cycle_length)
                # print(f"dbg elapsed time {time.time() - start_time}")
                all_cycles += selected_cycles
            
            return all_cycles
        
        def cluster_by_almost_full_cliques(working_graph, density_threshold=0.8):
            min_size = 2
            dense_subgraphs = []
            nodes = list(working_graph.get_nodes_ids())
            
            for size in range(min_size, len(nodes) + 1):
                for node_set in combinations(nodes, size):
                    subgraph = working_graph.graph.subgraph(node_set)
                    possible_edges = size * (size - 1) / 2  # Total edges in a complete working_graph
                    actual_edges = subgraph.number_of_edges()
                    if actual_edges / possible_edges >= density_threshold:
                        dense_subgraphs.append(frozenset(node_set))

            return dense_subgraphs
        
        def cluster_by_GMC(working_graph):
            if working_graph.graph.size() == 0 or len(working_graph.graph.edges()) == 0:
                return [{n} for n in working_graph.graph.nodes()]
            
            communities = list(greedy_modularity_communities(working_graph.graph))
            return [frozenset(c) for c in communities if len(set(c)) > 1]
        
        def cluster_by_ALC(working_graph):
            clusters = list(asyn_lpa_communities(working_graph.graph, seed=42))
            return [frozenset(c) for c in clusters]
        

        
        # filter_in_nodes = graph.filter_graph_by_degree_range([2,3,4,5,6])
        # self.logger.info(f'sbg filter_in_nodes {filter_in_nodes}')
        # start_time = time.time()
        # all_clusters = cluster_by_cycles(copy.deepcopy(graph))
        # cluster_by_cycles_time = time.time()
        # print(f"dbg cluster_by_cycles_time {cluster_by_cycles_time - start_time}")
        # all_clusters = cluster_by_almost_full_cliques(copy.deepcopy(graph), density_threshold=0.8)
        # cluster_by_almost_full_cliques_time = time.time()
        # print(f"dbg cluster_by_almost_full_cliques_time {cluster_by_almost_full_cliques_time - cluster_by_cycles_time}")
        all_clusters = cluster_by_GMC(filtered_graph)
        # cluster_by_GMC_time = time.time()
        # print(f"dbg cluster_by_GMC_time {cluster_by_GMC_time - cluster_by_almost_full_cliques_time}")
        # all_clusters = cluster_by_ALC(copy.deepcopy(graph))
        # cluster_by_ALC_time = time.time()
        # print(f"dbg cluster_by_ALC_time {cluster_by_ALC_time - cluster_by_GMC_time}")

        # self.logger.info(f"dbg  all_clusters {all_clusters}")
        selected_rooms_dicts = []
        if all_clusters:
            viz_values = {}
            
            colors = ["cyan", "orange", "purple", "magenta", "olive", "tan", "coral", "pink", "violet", "sienna", "yellow"]
            for i, cycle in enumerate(all_clusters):
                # room_dict = {"ws_ids": list(set(cycle))}
                # room_dict["ws_centers"] = [graph.get_attributes_of_node(node_id)["center"] for node_id in list(set(cycle))]
                for node_id in cycle:
                    viz_values.update({node_id: colors[i%len(colors)]})

                # if self.use_gnn_factors:
                #     planes_feats_6p = [np.concatenate([graph.get_attributes_of_node(node_id)["center"],graph.get_attributes_of_node(node_id)["normal"]/np.linalg.norm(graph.get_attributes_of_node(node_id)["normal"])]) for node_id in cycle]

                #     max_d = 20.
                #     centers = [graph.get_attributes_of_node(node_id)["center"][:2] / np.array([max_d, max_d]) for node_id in list(set(cycle))]

                #     planes_feats_4p = [self.correct_plane_direction(plane_6_params_to_4_params(plane_feats_6p)) / np.array([1, 1, 1, max_d]) for plane_feats_6p in planes_feats_6p]
                #     x = torch.cat((torch.tensor(planes_feats_4p).float(), torch.tensor(centers).float()), dim=1) ### TODO TODO TODO check if it still applies
                #     x = torch.cat((x,  torch.tensor([np.zeros(len(x[0]))])),dim=0).float()
                #     x1, x2 = [], []
                #     for i in range(x.size(0) - 1):
                #         x1.append(i)
                #         x2.append(x.size(0) - 1)
                #     edge_index = torch.tensor(np.array([x1, x2]).astype(np.int64))
                #     batch = torch.tensor(np.zeros(x.size(0)).astype(np.int64))
                #     nn_outputs = self.factor_nn.infer(x, edge_index, batch, "room").numpy()[0]
                #     center = np.array([nn_outputs[0], nn_outputs[1], 0]) * np.array([max_d, max_d, 1])
                # else:
                #     center = np.sum(np.stack([graph.get_attributes_of_node(node_id)["center"] for node_id in cycle]).astype(np.float32), axis = 0)/len(cycle)

                # tmp_i += 1
                # graph.add_nodes([(tmp_i,{"type" : "room","viz_type" : "Point", "viz_data" : center[:2],"center" : center, "viz_feat" : 'ro'})])
                
                # for node_id in list(set(cycle)):
                #     graph.add_edges([(tmp_i, node_id, {"type": "ws_belongs_room", "x": [], "viz_feat" : 'red', "linewidth":1.0, "alpha":0.5})])

                # room_dict["center"] = center
                # selected_rooms_dicts.append(room_dict)
            graph.set_node_attributes("viz_feat", viz_values)
            # time.sleep(565)
            # if self.settings["report"]["save"]:
            #     plt.savefig(os.path.join(self.report_path,f'room clustering.png'), bbox_inches='tight')
                
        graph = graph.filter_graph_by_edge_attributes({"type":"ws_belongs_room"})

        return all_clusters, graph
    
    def cluster_walls(self, graph):
        graph = copy.deepcopy(graph)
        graph = graph.filter_graph_by_edge_attributes({"type":"ws_same_wall"})
        graph.to_undirected(type= "smooth")
        
        all_edges = list(graph.get_edges_ids())
        all_edges = [set(edge) for edge in all_edges]
        colors = ["cyan", "orange", "purple", "magenta", "olive", "tan", "coral", "pink", "violet", "sienna", "yellow"]
        viz_values = {}
        edges_dicst = []
        tmp_i = 200
        for i, edge in enumerate(all_edges):
            wall_dict = {"ws_ids": list(set(edge))}
            # wall_dict["ws_centers"] = [graph.get_attributes_of_node(node_id)["center"] for node_id in list(set(edge))]
            for node_id in edge:
                viz_values.update({node_id: colors[i%len(colors)]})
            # planes_centers = np.array([np.array(graph.get_attributes_of_node(node_id)["center"]) for node_id in edge])
            
            # if self.use_gnn_factors:
            #     max_d = 20.
            #     planes_centers_normalized = np.array([np.array(graph.get_attributes_of_node(node_id)["center"]) / np.array([max_d, max_d, max_d]) for node_id in edge])
            #     planes_feats_6p = [np.concatenate([graph.get_attributes_of_node(node_id)["center"],graph.get_attributes_of_node(node_id)["normal"]]) for node_id in edge]
            #     planes_feats_4p = np.array([self.correct_plane_direction(plane_6_params_to_4_params(plane_feats_6p)) / np.array([1, 1, 1, max_d]) for plane_feats_6p in planes_feats_6p])
            #     planes_feats_4p = torch.tensor(planes_feats_4p, dtype=torch.float32) if isinstance(planes_feats_4p, np.ndarray) else planes_feats_4p
            #     x = torch.cat((torch.tensor(planes_centers_normalized, dtype=torch.float32), 
            #                 planes_feats_4p[:, :3].float()), dim=1)
            #     zeros_row = torch.zeros(1, x.size(1), dtype=torch.float32)  # REMOVE THIS FROM F-GNN architecture
            #     x = torch.cat((x, zeros_row), dim=0)
            #     x1, x2 = [], []
            #     for i in range(x.size(0) - 1):
            #         x1.append(i)
            #         x2.append(x.size(0) - 1)
            #     edge_index = torch.tensor(np.array([x1, x2]).astype(np.int64))
            #     batch = torch.tensor(np.zeros(x.size(0)).astype(np.int64))
            #     nn_outputs = self.factor_nn.infer(x, edge_index, batch, "wall").numpy()[0]
            #     center = np.array([nn_outputs[0], nn_outputs[1], 0]) * np.array([max_d, max_d, 1])
            # else:
            #     center = np.sum(np.stack([graph.get_attributes_of_node(node_id)["center"] for node_id in edge]).astype(np.float32), axis = 0)/len(edge)
            #     wall_points = [center, center]

            # graph.add_nodes([(tmp_i,{"type" : "wall","viz_type" : "Point","center" : center, "viz_data" : center, "viz_feat" : 'mo'})])
            # for node_id in list(edge):
            #     graph.add_edges([(tmp_i, node_id, {"type": "ws_belongs_wall", "x": [], "viz_feat" : 'm', "linewidth":1.0, "alpha":0.5})])

            # tmp_i += 1
            # wall_dict["center"] = center
            # wall_dict["wall_points"] = planes_centers
            
            edges_dicst.append(wall_dict)
        graph.set_node_attributes("viz_feat", viz_values)
        visualize_nxgraph(graph, image_name = "wall clustering", include_node_ids= False)
        # if self.settings["report"]["save"]:
        #     plt.savefig(os.path.join(self.report_path,f'wall clustering.png'), bbox_inches='tight')
        return all_edges, graph

    def cluster_floors(self, graph):
        graph = copy.deepcopy(graph)
        room_nodes_ids = graph.filter_graph_by_node_attributes({"type": "room"}).get_nodes_ids()
        graph.to_undirected(type= "smooth")
        
        floor_node_id = 500
        rooms_dicts = []
        
        if self.use_gnn_factors:
            max_d = 20.
            planes_centers = np.array([np.array(graph.get_attributes_of_node(node_id)["center"]) / np.array([max_d, max_d, max_d]) for node_id in room_nodes_ids])
            x = torch.cat((torch.tensor(planes_centers).float(),  torch.tensor([np.zeros(len(planes_centers[0]))])),dim=0).float()
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
        
        # if self.settings["report"]["save"]:
        #     plt.savefig(os.path.join(self.report_path,f'wall clustering.png'), bbox_inches='tight')
        return rooms_dicts
    

    def cluster_RoomWall(self, graph, mode):
        clusters = {}
        clusters["room"], rooms_graph = self.cluster_rooms(copy.deepcopy(graph))
        room_fig = visualize_nxgraph(rooms_graph, image_name = f"{mode} Inference rooms graph", include_node_ids= False)
        self.graphs_subplot.update_plot_with_figure(f"{mode} Inference rooms graph", room_fig, square_it = True)
        plt.close(room_fig)
        clusters["wall"], walls_graph = self.cluster_walls(graph)
        wall_fig = visualize_nxgraph(walls_graph, image_name = f"{mode} Inference walls graph", include_node_ids= False)
        self.graphs_subplot.update_plot_with_figure(f"{mode} Inference walls graph", wall_fig, square_it = True)
        plt.close(wall_fig)

        return clusters, graph
    

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

    def free_gpu_memory(self):
        self.model.to('cpu')
        del self.hdataset
        del self.optimizer
        del self.metric_values
        torch.cuda.empty_cache()
        gc.collect()
