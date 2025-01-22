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

# from .FactorNN import FactorNN

# graph_reasoning_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_reasoning")
# sys.path.append(graph_reasoning_dir)
# from graph_reasoning.from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata
# graph_datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets")
# sys.path.append(graph_datasets_dir)
# graph_matching_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_matching")
# sys.path.append(graph_matching_dir)
from graph_matching.utils import plane_6_params_to_4_params
# graph_factor_nn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_factor_nn")
# sys.path.append(graph_factor_nn_dir)
from graph_factor_nn.FactorNNBridge import FactorNNBridge
from graph_reasoning.from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata
from graph_reasoning.MetricsSubplot import MetricsSubplot
from graph_reasoning.GNNs.v1.G_GNNv1 import G_GNNv1
from graph_reasoning.GNNs.v2.G_GNNv2 import G_GNNv2
from graph_datasets.graph_visualizer import visualize_nxgraph


class GNNWrapper():
    def __init__(self, settings, report_path, logger = None, ID=0) -> None:
        print(f"GNNWrapper{ID}: ", Fore.BLUE + "Initializing" + Fore.WHITE)
        self.settings = settings
        self.target_concept = settings["report"]["target_concept"]
        self.report_path = report_path
        self.logger = logger
        self.ID = ID
        self.use_gnn_factors = False
        self.pth_path = os.path.join(self.report_path,'model.pth')
        self.best_pth_path = os.path.join(self.report_path,'model_best.pth')
        self.set_cuda_device()

        if logger:
            self.logger.info(f"GNNWrapper{self.ID} : torch device => {self.device}")
        else:
            print(f"GNNWrapper{self.ID}: ", Fore.BLUE + f"Torch device => {self.device}" + Fore.WHITE)
        metric_values_dict = {"loss_avg" : [], "auc" : [], "acc" : [], "prec" : [], "rec" : [], "f1" : [], "pred_pos_rate" : [], "gt_pos_rate": [], "score":[]}
        self.metric_values = {"train" : copy.deepcopy(metric_values_dict), "val" : copy.deepcopy(metric_values_dict),\
                            "test" : copy.deepcopy(metric_values_dict), "inference" : copy.deepcopy(metric_values_dict),}
        if self.use_gnn_factors:
            self.factor_nn = FactorNNBridge(["room", "wall", "floor"])

        plot_names_map = {"train Metrics": 0, "train RoomWall inference": 1, "train Inference rooms graph": 2,"train Inference walls graph": 3,\
                          "val Metrics": 4, "val RoomWall inference": 5, "val Inference rooms graph": 6,"val Inference walls graph": 7,\
                          "test Metrics": 8, "test RoomWall inference": 9,"test Inference rooms graph": 10,"test Inference walls graph":11}

        self.metric_subplot = MetricsSubplot(name=str(ID), nrows=3, ncols=4, plot_names_map=plot_names_map)

    def set_cuda_device(self):
        min_free_memory_mb = 1024
        if self.settings["gnn"]["use_cuda"]:
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                available_devices = []

                # Check each GPU's memory
                for device_id in range(num_devices):
                    free_memory, total_memory = torch.cuda.mem_get_info(device_id)
                    free_memory_mb = free_memory / (1024 * 1024)  # Convert to MB
                    total_memory_mb = total_memory / (1024 * 1024)  # Convert to MB
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

    # Example usage
    # hdata_list: List of PyTorch Geometric HeteroData objects
            
    def define_GCN(self):
        print(f"GNNWrapper{self.ID}: ", Fore.BLUE + "Defining GCN" + Fore.WHITE)
            
        self.model = G_GNNv2(self.settings, self.logger)

    def train(self, verbose = False):
        print(f"GNNWrapper{self.ID}: ", Fore.BLUE + "Training" + Fore.WHITE)

        best_val_loss = float('inf')
        training_settings = self.settings["training"]
        patience = training_settings["patience"]
        trigger_times = 0

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.settings["gnn"]["lr"])
        self.criterion = torch.nn.CrossEntropyLoss()
        original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]

        for epoch in (pbar := tqdm.tqdm(range(1, training_settings["epochs"]), colour="blue")):
            pbar.set_description(f"Epoch{self.ID}")
            total_loss = total_examples = 0
            gt_in_train_dataset, probs_in_train_dataset = [], []
            start_time = time.time()
            for i, hdata in enumerate(self.hdataset["train"]):
                self.optimizer.zero_grad()
                # sub_start_time = time.time()
                hdata.to(self.device)
                # sub_part_1_end = time.time()
                logits = self.model(hdata.x_dict, hdata.edge_index_dict,hdata.edge_label_dict)
                # sub_part_2_end = time.time()
                gt = hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label.to(self.device)[hdata.edge_label_dict["edge_index_to_edge_label_index"]]
                gt_in_train_dataset = gt_in_train_dataset + list(gt.cpu().numpy())
                loss = self.criterion(logits, gt)
                # sub_part_3_end = time.time()
                loss.backward()
                # sub_part_4_end = time.time()
                self.optimizer.step()
                # sub_part_5_end = time.time()
                total_loss += float(loss) * logits.numel()
                total_examples += logits.numel()
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                probs_in_train_dataset = probs_in_train_dataset + [list(probs)]
                preds = np.argmax(probs, axis=1)
                sub_part_6_end = time.time()
                if i == len(self.hdataset["train"]) - 1:
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
            probs_in_train_dataset = np.concatenate(probs_in_train_dataset, axis=0)
            # print(f"dbg len(probs_in_train_dataset) {probs_in_train_dataset[:50]}")
            accuracy, precission, recall, f1, auc = self.compute_metrics_from_all_predictions(gt_in_train_dataset, probs_in_train_dataset, verbose= False)
            loss_avg = total_loss / total_examples
            score = (0.5*loss_avg + 0.5*(1-auc))

            self.metric_values["train"]["auc"].append(auc)
            self.metric_values["train"]["acc"].append(accuracy)
            self.metric_values["train"]["prec"].append(precission)
            self.metric_values["train"]["rec"].append(recall)
            self.metric_values["train"]["f1"].append(f1)
            # self.metric_values["train"]["gt_pos_rate"].append(gt_pos_rate)
            # self.metric_values["train"]["pred_pos_rate"].append(pred_pos_rate)
            self.metric_values["train"]["loss_avg"].append(loss_avg)
            self.metric_values["train"]["score"].append(score)

            if verbose:
                ### Metrics
                self.plot_metrics("train", metrics= ["loss_avg", "acc", "prec", "rec", "f1", "auc", "score"])
                # if self.settings["report"]["save"]:
                #     plt.savefig(os.path.join(self.report_path,f'train_metrics.png'), bbox_inches='tight')

                ### inference - Inference
                merged_graph = self.merge_predicted_edges(copy.deepcopy(self.nxdataset["train"][-1]), predicted_edges_last_graph)
                fig = visualize_nxgraph(merged_graph, image_name = f"train {self.target_concept} inference", include_node_ids= False)
                self.metric_subplot.update_plot_with_figure(f"train {self.target_concept} inference", fig, square_it = True)

                if self.target_concept == "RoomWall":
                    clusters, inferred_graph = self.cluster_RoomWall(merged_graph, "train")
            # part_2_end = time.time()
            # print(f"Training Part 1 took {part_1_end - start_time:.4f} seconds.")
            # print(f"Training Part 2 took {part_2_end - part_1_end:.4f} seconds.")
            # train_end_time = time.time()
            val_loss = self.validate("val", verbose)
            # val_end_time = time.time()
            # print(f"Training took {train_end_time - start_time:.4f} seconds.")
            # print(f"Validation took {val_end_time - train_end_time:.4f} seconds.")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
                self.best_model = copy.deepcopy(self.model)
                self.save_best_model()
            else:
                trigger_times += 1
            if trigger_times >= patience:
                print(f"GNNWrapper{self.ID}: ", Fore.BLUE + "Early stopping triggered" + Fore.WHITE)
                # test_loss = self.validate( "test", verbose= True)
                break

            self.save_best_model()

        self.metric_subplot.close()
        self.model = self.best_model
        test_score = self.validate( "test", verbose= True)
        return test_score


    def validate(self,tag,verbose = False):
        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]
        original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        self.model = self.model.to(self.device)
        total_loss = total_examples = 0
        gt_in_val_dataset, probs_in_val_dataset = [], []

        for i, hdata in enumerate(self.hdataset[tag]):
            total_loss = total_examples = 0

            with torch.no_grad():
                hdata.to(self.device)

                logits = self.model(hdata.x_dict, hdata.edge_index_dict, hdata.edge_label_dict)
                gt = hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label.to(self.device)[hdata.edge_label_dict["edge_index_to_edge_label_index"]]
                gt_in_val_dataset = gt_in_val_dataset + list(gt.cpu().numpy())
                loss = self.criterion(logits, gt)
                total_loss += float(loss) * logits.numel()
                total_examples += logits.numel()
                
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                probs_in_val_dataset = probs_in_val_dataset + [list(probs)]
                preds = np.argmax(probs, axis=1)

            if i == len(self.hdataset[tag]) - 1:
                color_code = ["black", "blue", "orange"]
                predicted_edges_last_graph = [(ei[0], ei[1], {"type" : original_edge_types[preds[i]],\
                                            "label": preds[i], "viz_feat": color_code[preds[i]], "linewidth":0.5 if preds[i]==0 else 1.5,\
                                            "alpha":0.3 if preds[i]==0 else 1.}) for i, ei in enumerate(hdata.edge_label_dict["edge_label_index_tuples_compressed"])]
            
        probs_in_val_dataset = np.concatenate(probs_in_val_dataset, axis=0)
        accuracy, precission, recall, f1, auc = self.compute_metrics_from_all_predictions(gt_in_val_dataset, probs_in_val_dataset, verbose= False)
        loss_avg = total_loss / total_examples
        score = (0.5*loss_avg + 0.5*(1-auc))

        self.metric_values[tag]["auc"].append(auc)
        self.metric_values[tag]["acc"].append(accuracy)
        self.metric_values[tag]["prec"].append(precission)
        self.metric_values[tag]["rec"].append(recall)
        self.metric_values[tag]["f1"].append(f1)
        self.metric_values[tag]["loss_avg"].append(loss_avg)
        self.metric_values[tag]["score"].append(score)

        if verbose:
            ### Metrics
            self.plot_metrics(tag, metrics= ["acc", "prec", "rec", "f1", "auc", "loss_avg", "score"])

            ### inference - Inference
            merged_graph = self.merge_predicted_edges(copy.deepcopy(self.nxdataset[tag][-1]), predicted_edges_last_graph)
            fig = visualize_nxgraph(merged_graph, image_name = f"{tag} {self.target_concept} inference", include_node_ids= False)
            self.metric_subplot.update_plot_with_figure(f"{tag} {self.target_concept} inference", fig, square_it = True)
            del fig

            if self.target_concept == "RoomWall":
                clusters, inferred_graph = self.cluster_RoomWall(merged_graph, tag)

            if self.settings["report"]["save"]:
                self.metric_subplot.save(os.path.join(self.report_path,f'{self.target_concept} subplot.png'))
            
                # plt.savefig(os.path.join(self.report_path,f'{self.target_concept} subplot.png'), bbox_inches='tight')


        return score


    def infer(self, nx_data, verbose, use_gt = False):

        plot_names_map = {"infer RoomWall inference": 0, "infer Inference rooms graph": 1,"infer Inference walls graph": 2}

        self.metric_subplot = MetricsSubplot("infer", nrows=1, ncols=3, plot_names_map=plot_names_map)
    
        original_edge_types = ["None"] + [e[1] for e in self.settings["hdata"]["edges"]]
        color_code = ["black", "blue", "brown"]

        edge_types = [tuple((e[0],"training",e[2])) for e in self.settings["hdata"]["edges"]][0]
        self.model.encoder.training = False
        self.model = self.model.to(self.device)
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
                
            logits = self.model(hdata.x_dict, hdata.edge_index_dict, edge_label_dict)

            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            edge_index = list(hdata[edge_types[0],edge_types[1],edge_types[2]].edge_index.cpu().numpy())
            edge_index = np.array(list(zip(edge_index[0], edge_index[1])))
            if use_gt:
                preds = hdata[edge_types[0],edge_types[1],edge_types[2]].edge_label.to(self.device).cpu().numpy()

            predicted_edges_last_graph = [(ei[0], ei[1], {"type" : original_edge_types[preds[i]],\
                                        "label": preds[i], "viz_feat": color_code[preds[i]], "linewidth":0.5 if preds[i]==0 else 1.5,\
                                        "alpha":0.3 if preds[i]==0 else 1.}) for i, ei in enumerate(edge_label_index_tuples_compressed)]
            
            merged_graph = self.merge_predicted_edges(copy.deepcopy(nx_data), predicted_edges_last_graph)
            fig = visualize_nxgraph(merged_graph, image_name = f"infer {self.target_concept} inference")
            self.metric_subplot.update_plot_with_figure(f"infer {self.target_concept} inference", fig, square_it = True)
            
            if self.target_concept == "RoomWall":
                clusters, inferred_graph = self.cluster_RoomWall(merged_graph, "infer")
                
            if self.settings["report"]["save"]:
                self.metric_subplot.save(os.path.join(self.report_path,f'{self.target_concept} subplot.png'))
            

        return clusters


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
    

    def get_message_sharing_edges(self, nx_data):
        pass # To Be Rebuilt


    def merge_predicted_edges(self, unparented_base_graph, predictions):
        unparented_base_graph = copy.deepcopy(unparented_base_graph)
        unparented_base_graph.remove_all_edges()
        unparented_base_graph.add_edges(predictions)
        return unparented_base_graph
    

    def plot_metrics(self, tag, metrics):
        fig = plt.figure(f"{self.report_path} Metrics")
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_ylim([0, 1])
        label_mapping = {"acc": "Accuracy", "prec":"Precission", "rec":"Recall", "f1":"F1", "auc":"AUC", "loss_avg":"Loss avg", "score":"Score"}
        color_mapping = {"acc": "orange", "prec":"green", "rec":"red", "f1":"purple", "auc":"brown", "loss_avg":"blue", "score":"black"}
        for metric_name in metrics:
            ax.plot(np.array(self.metric_values[tag][metric_name]), label = label_mapping[metric_name], color = color_mapping[metric_name])
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Rate')
        self.metric_subplot.update_plot_with_figure(f"{tag} Metrics", fig, square_it = False)
        plt.close(fig)
        

    def cluster_rooms(self, old_graph):
        graph = copy.deepcopy(old_graph)
        graph = graph.filter_graph_by_edge_attributes({"type":"ws_same_room"})
        graph.to_undirected(type= "smooth")
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
            return [frozenset(c) for c in communities]
        
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
        all_clusters = cluster_by_GMC(copy.deepcopy(graph))
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
            tmp_i = 100
            for i, cycle in enumerate(all_clusters):
                room_dict = {"ws_ids": list(set(cycle))}
                room_dict["ws_centers"] = [graph.get_attributes_of_node(node_id)["center"] for node_id in list(set(cycle))]
                for node_id in cycle:
                    viz_values.update({node_id: colors[i%len(colors)]})

                if self.use_gnn_factors:
                    planes_feats_6p = [np.concatenate([graph.get_attributes_of_node(node_id)["center"],graph.get_attributes_of_node(node_id)["normal"]/np.linalg.norm(graph.get_attributes_of_node(node_id)["normal"])]) for node_id in cycle]

                    max_d = 20.
                    centers = [graph.get_attributes_of_node(node_id)["center"][:2] / np.array([max_d, max_d]) for node_id in list(set(cycle))]

                    planes_feats_4p = [self.correct_plane_direction(plane_6_params_to_4_params(plane_feats_6p)) / np.array([1, 1, 1, max_d]) for plane_feats_6p in planes_feats_6p]
                    x = torch.cat((torch.tensor(planes_feats_4p).float(), torch.tensor(centers).float()), dim=1)
                    x = torch.cat((x,  torch.tensor([np.zeros(len(x[0]))])),dim=0).float()
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
                graph.add_nodes([(tmp_i,{"type" : "room","viz_type" : "Point", "viz_data" : center[:2],"center" : center, "viz_feat" : 'ro'})])
                
                for node_id in list(set(cycle)):
                    graph.add_edges([(tmp_i, node_id, {"type": "ws_belongs_room", "x": [], "viz_feat" : 'red', "linewidth":1.0, "alpha":0.5})])

                room_dict["center"] = center
                selected_rooms_dicts.append(room_dict)
            graph.set_node_attributes("viz_feat", viz_values)
            # time.sleep(565)
            # if self.settings["report"]["save"]:
            #     plt.savefig(os.path.join(self.report_path,f'room clustering.png'), bbox_inches='tight')
                
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
                planes_feats_4p = torch.tensor(planes_feats_4p, dtype=torch.float32) if isinstance(planes_feats_4p, np.ndarray) else planes_feats_4p
                x = torch.cat((torch.tensor(planes_centers_normalized, dtype=torch.float32), 
                            planes_feats_4p[:, :3].float()), dim=1)
                zeros_row = torch.zeros(1, x.size(1), dtype=torch.float32)  # REMOVE THIS FROM F-GNN architecture
                x = torch.cat((x, zeros_row), dim=0)
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

            graph.add_nodes([(tmp_i,{"type" : "wall","viz_type" : "Point","center" : center, "viz_data" : center, "viz_feat" : 'mo'})])
            for node_id in list(edge):
                graph.add_edges([(tmp_i, node_id, {"type": "ws_belongs_wall", "x": [], "viz_feat" : 'm', "linewidth":1.0, "alpha":0.5})])

            tmp_i += 1
            wall_dict["center"] = center
            wall_dict["wall_points"] = planes_centers
            
            edges_dicst.append(wall_dict)
        graph.set_node_attributes("viz_feat", viz_values)
        visualize_nxgraph(graph, image_name = "wall clustering", include_node_ids= False)
        # if self.settings["report"]["save"]:
        #     plt.savefig(os.path.join(self.report_path,f'wall clustering.png'), bbox_inches='tight')
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
        self.metric_subplot.update_plot_with_figure(f"{mode} Inference rooms graph", room_fig, square_it = True)
        plt.close(room_fig)
        clusters["wall"], walls_graph = self.cluster_walls(graph)
        wall_fig = visualize_nxgraph(walls_graph, image_name = f"{mode} Inference walls graph", include_node_ids= False)
        self.metric_subplot.update_plot_with_figure(f"{mode} Inference walls graph", wall_fig, square_it = True)
        plt.close(wall_fig)

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

    def free_gpu_memory(self):
        self.model.to('cpu')
        del self.optimizer
        torch.cuda.empty_cache()
        gc.collect()
