import matplotlib.pyplot as plt
import json, os, time, shutil, sys, copy
import optuna
import torch

from GNNWrapper import GNNWrapper
from graph_datasets.graph_visualizer import visualize_nxgraph

# synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"situational_graphs_datasets", "graph_datasets")
# sys.path.append(synthetic_datset_dir)
from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_datasets.config import get_config as get_datasets_config
from graph_reasoning.config import get_config as get_reasoning_config

class InferenceTest():
    def __init__(self):
        self.target_concept = "RoomWall"

        self.synteticdataset_settings = get_datasets_config("graph_reasoning")
        self.synteticdataset_settings["base_graphs"]["n_buildings"] = 1
        self.synteticdataset_settings["training_split"]["val"] = 0.0
        self.synteticdataset_settings["training_split"]["test"] = 0.0
        self.graph_reasoning_settings_base = get_reasoning_config(f"same_{self.target_concept}_training")
        self.graph_reasoning_settings = self.graph_reasoning_settings_base
        self.prepare_report_folder()
        self.prepare_dataset()
        self.prepare_gnn()

    def prepare_report_folder(self):
        self.report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"reports","synthetic",self.graph_reasoning_settings_base["report"]["name"], "_inference")
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
        
        combined_settings = {"dataset": self.synteticdataset_settings, "graph_reasoning": self.graph_reasoning_settings_base}
        with open(os.path.join(self.report_path, "settings.json"), "w") as fp:
            json.dump(combined_settings, fp)
        
        
    def prepare_dataset(self):
        dataset_generator = SyntheticDatasetGenerator(self.synteticdataset_settings, None, self.report_path)
        dataset_generator.create_dataset()
        settings_hdata = self.graph_reasoning_settings_base["hdata"]
        filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["noise"]
        extended_nxdatset = dataset_generator.extend_nxdataset(filtered_nxdataset, "training", "training")
        self.normalized_nxdatset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)
        self.normalized_nxdatset["inference"] = self.normalized_nxdatset["train"] + self.normalized_nxdatset["val"] + self.normalized_nxdatset["test"]
        del self.normalized_nxdatset["train"]
        del self.normalized_nxdatset["val"]
        del self.normalized_nxdatset["test"]

    def prepare_gnn(self):
        self.gnn_wrapper = GNNWrapper(self.graph_reasoning_settings, self.report_path)
        self.gnn_wrapper.define_GCN()
        self.gnn_wrapper.set_dataset(self.normalized_nxdatset)

    def inference_test(self):
        for nx_data in inference_test.normalized_nxdatset["inference"]:
            self.gnn_wrapper.infer(nx_data, True, use_gt=False)
            input("Press Enter to continue...")
    

inference_test = InferenceTest()
inference_test.inference_test()
print(f"dbg flaaaaaaaaaaaaaaaaaaaaaaaaaaag")
inference_test.prepare_gnn()
inference_test.inference_test()