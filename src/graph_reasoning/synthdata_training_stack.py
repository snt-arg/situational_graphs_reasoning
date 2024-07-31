import matplotlib.pyplot as plt
import json, os, time, shutil, sys

from GNNWrapper import GNNWrapper
from graph_datasets.graph_visualizer import visualize_nxgraph

# synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"situational_graphs_datasets", "graph_datasets")
# sys.path.append(synthetic_datset_dir)
from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_datasets.config import get_config as get_datasets_config
from graph_reasoning.config import get_config as get_reasoning_config

class GraphReasoning():
    def __init__(self):
        target_concept = "room"
        # with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"config", f"same_{target_concept}_training.json")) as f:
        self.graph_reasoning_settings = get_reasoning_config(f"same_{target_concept}_training")
        # with open(os.path.join(os.path.dirname(synthetic_datset_dir),"config", "graph_reasoning.json")) as f:
        self.synteticdataset_settings = get_datasets_config("graph_reasoning")
    
    def train_stack(self):
        self.prepare_report_folder()
        self.prepare_dataset()
        self.prepare_gnn()
        self.train()

    def prepare_report_folder(self):
        self.report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"reports","synthetic",self.graph_reasoning_settings["report"]["name"])
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
        settings_hdata = self.graph_reasoning_settings["hdata"]
        filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"][0])["noise"]
        extended_nxdatset = dataset_generator.extend_nxdataset(filtered_nxdataset, settings_hdata["edges"][0][1], "training")
        self.normalized_nxdatset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)

    def prepare_gnn(self):
        self.gnn_wrapper = GNNWrapper(self.graph_reasoning_settings, self.report_path)
        self.gnn_wrapper.define_GCN()
        self.gnn_wrapper.set_dataset(self.normalized_nxdatset)

    def train(self):
        self.gnn_wrapper.train(verbose= True)

gr = GraphReasoning()
gr.train_stack()