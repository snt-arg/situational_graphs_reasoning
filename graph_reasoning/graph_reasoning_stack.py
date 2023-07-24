import matplotlib.pyplot as plt
import json, os, time, shutil, sys

from GNNWrapper import GNNWrapper

synthetic_datset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_datasets", "graph_datasets")
sys.path.append(synthetic_datset_dir)
from SyntheticDatasetGenerator import SyntheticDatasetGenerator

class GraphReasoning():
    def __init__(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"config", "same_room_training.json")) as f:
            self.graph_reasoning_settings = json.load(f)
        with open(os.path.join(os.path.dirname(synthetic_datset_dir),"config", "graph_reasoning.json")) as f:
            self.synteticdataset_settings = json.load(f)
    
        self.prepare_report_folder()
        self.prepare_dataset()
        self.prepare_gnn()
        self.train()

    def prepare_report_folder(self):
        self.report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"reports",self.graph_reasoning_settings["report"]["name"])
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
        dataset_generator = SyntheticDatasetGenerator(self.synteticdataset_settings)
        filtered_nxdataset = dataset_generator.get_filtered_datset(["ws"],["ws_same_room"])["original"]
        self.extended_nxdatset = dataset_generator.extend_nxdataset(filtered_nxdataset)

    def prepare_gnn(self):
        self.gnn_wrapper = GNNWrapper(self.extended_nxdatset, self.graph_reasoning_settings, self.report_path)
        self.gnn_wrapper.define_GCN()

    def train(self):
        self.gnn_wrapper.train(verbose= True)

    def final_inference(self):
        predicted_edges = self.gnn_wrapper.infer(self.extended_nxdatset["inference"], True) ### TODO datset already saved in the gnnwrapper

GraphReasoning()