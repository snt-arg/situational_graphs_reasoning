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

class GNNTrainer():
    def __init__(self):
        self.target_concept = "RoomWall"

        self.synteticdataset_settings = get_datasets_config("graph_reasoning")
        self.graph_reasoning_settings_base = get_reasoning_config(f"same_{self.target_concept}_training")
        self.prepare_report_folder()
        self.prepare_dataset()
        self.set_hyperparameters_mappings()

    # def train_stack(self):
    #     self.prepare_gnn()
    #     self.train()

    def prepare_report_folder(self):
        self.report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"reports","synthetic",self.graph_reasoning_settings_base["report"]["name"])
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

    def prepare_gnn(self):
        self.gnn_wrapper = GNNWrapper(self.graph_reasoning_settings, self.report_path)
        self.gnn_wrapper.define_GCN()
        self.gnn_wrapper.set_dataset(self.normalized_nxdatset)

    def set_hyperparameters_mappings(self):
        d = {}
        # d.update({"decoder_hc" : ["gnn", "decoder", "hidden_channels"]})
        d.update({"lr" : ["gnn", "lr"]})
        d.update({"enc_nod_hc" : ["gnn", "encoder", "nodes", "hidden_channels"]})
        d.update({"enc_edg_hc" : ["gnn", "encoder", "edges", "hidden_channels"]})
        d.update({"dec_hc_0" : ["gnn", "decoder", "hidden_channels", 0]})
        d.update({"dec_hc_1" : ["gnn", "decoder", "hidden_channels", 1]})
        self.hyperparameters_mappings = d

    def objective(self, trial):
        # Suggest hyperparameters to optimize
        hyperparameters_values = {}
        hyperparameters_values['lr'] = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
        # hyperparameters_values['num_layers'] = trial.suggest_int('num_layers', 1, 5)
        hyperparameters_values['enc_nod_hc'] = trial.suggest_int('enc_nod_hc', 8, 256)
        hyperparameters_values['enc_edg_hc'] = trial.suggest_int('enc_edg_hc', 8, 256)
        hyperparameters_values['dec_hc_0'] = trial.suggest_int('dec_hc_0', 8, 256)
        hyperparameters_values['dec_hc_1'] = trial.suggest_int('dec_hc_1', 8, 256)

        self.graph_reasoning_settings = self.update_settings_dict(self.graph_reasoning_settings_base, self.hyperparameters_mappings, hyperparameters_values)
        self.prepare_gnn()
        score = self.train()
        trial.set_user_attr("model", copy.deepcopy(self.gnn_wrapper))
        return score

    def hyperparameters_optimization(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=50)
        best_hyperparameters = study.best_params
        best_model = study.best_trial.user_attrs["model"]
        best_graph_reasoning_settings = self.update_settings_dict(self.graph_reasoning_settings_base, self.hyperparameters_mappings, best_hyperparameters)
        with open(os.path.join(self.report_path, f"same_{self.target_concept}_best.json"), "w") as fp:
            json.dump(best_graph_reasoning_settings, fp)
        best_model.save_model(os.path.join(self.report_path, f"model_{self.target_concept}_best.pth"))   

    def train(self):
        score = self.gnn_wrapper.train(verbose= True)
        return score
    
    def update_settings_dict(self, base_settings, mappings, values_dict):

        def update_nested_dict(d, keys, value):
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value

        new_settings = copy.deepcopy(base_settings)
        for key in mappings.keys():
            mapping = mappings[key]
            update_nested_dict(new_settings, mapping, values_dict[key])

        return new_settings
    

gnn_trainer = GNNTrainer()
gnn_trainer.hyperparameters_optimization()