from torch_geometric.datasets import ZINC
from tqdm import tqdm
import torch
from torch import nn
import math
import wandb
import os, json, sys, shutil


# from graph_datasets.graph_visualizer import visualize_nxgraph

synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),"graph_datasets", "graph_datasets")
sys.path.append(synthetic_datset_dir)
from SyntheticDatasetGenerator import SyntheticDatasetGenerator

from models import DiffusionOrderingNetwork, DenoisingNetwork
from grapharm import GraphARM
from grapharm import GraphARM

class GraphReasoning():
    def __init__(self):
        # target_concept = "room"
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"config", f"grapharm_training.json")) as f:
            self.graph_reasoning_settings = json.load(f)
        with open(os.path.join(os.path.dirname(synthetic_datset_dir),"config", "graph_reasoning_grapharm.json")) as f:
            self.synteticdataset_settings = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device {self.device}")

    
    def train_stack(self):
        self.prepare_report_folder()
        self.prepare_dataset()
        self.prepare_grapharm()
        self.train()

    def prepare_report_folder(self):
        self.report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"reports","synthetic",self.graph_reasoning_settings["report"]["name"])
        if not os.path.exists(self.report_path):
            os.makedirs(self.report_path)
        else:
            for filename in os.listdir(self.report_path):
                file_path = os.path.join(self.report_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        
        combined_settings = {"dataset": self.synteticdataset_settings, "graph_reasoning": self.graph_reasoning_settings}
        with open(os.path.join(self.report_path, "settings.json"), "w") as fp:
            json.dump(combined_settings, fp)

        
    def prepare_dataset(self):
        dataset_generator = SyntheticDatasetGenerator(self.synteticdataset_settings, None, self.report_path)
        dataset_generator.create_dataset()
        settings_hdata = self.graph_reasoning_settings["semantics"]["all"]
        filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["original"]
        extended_nxdatset = dataset_generator.extend_nxdataset(filtered_nxdataset, "k_nearest", "training")
        self.dataset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)

        # self.dataset = dataset_generator.dataset_to_hdata(self.dataset)
        # print(f"flag self.dataset {self.dataset['train'][0]['room','ws_belongs_room','ws'].edge_index}")
        # asfd

    def prepare_grapharm(self):
        semantics_settings = self.graph_reasoning_settings["semantics"]
        don_settings = self.graph_reasoning_settings["neural_networks"]["DON"]
        diff_ord_net = DiffusionOrderingNetwork(don_settings, semantics_settings,device=self.device)

        deno_settings = self.graph_reasoning_settings["neural_networks"]["deno"]
        denoising_net = DenoisingNetwork(deno_settings, semantics_settings, device=self.device)

        wandb.init(
                project="GraphARM original",
                group=f"v2.3.1",
                name=f"local",
                config={
                    "policy": "train",
                    "n_epochs": 10000,
                    "batch_size": 1,
                    "lr": 1e-3,
                },
                # mode='disabled'
        )

        torch.autograd.set_detect_anomaly(True)


        self.grapharm = GraphARM(
            dataset=self.dataset,
            denoising_network=denoising_net,
            diffusion_ordering_network=diff_ord_net,
            semantics_settings = semantics_settings,
            device=self.device
        )

    def train(self):
        batch_size = self.graph_reasoning_settings["training"]["batch"]
        try:
            self.grapharm.load_model()
            print("Loaded model")
        except:
            print ("No model to load")
        # train loop
        for epoch in range(self.graph_reasoning_settings["training"]["epochs"]):
            print(f"Epoch {epoch}")
            self.grapharm.train_step(
                train_don_data=self.dataset["train"][int((2*epoch*batch_size)%len(self.dataset["train"])):int(((2*epoch + 1)*batch_size)%len(self.dataset["train"]))], 
                train_deno_data=self.dataset["val"][int(((2*epoch + 1)*batch_size)%len(self.dataset["val"])):int((batch_size*(2*epoch + 2))%len(self.dataset["val"]))],
                val_data = [self.dataset["test"][0]],
                M=4
            )
            self.grapharm.save_model()

gr = GraphReasoning()
gr.train_stack()

# instanciate the dataset
# dataset = ZINC(root='~/workspace/GraphDiffusionImitate/data/ZINC', transform=None, pre_transform=None)

