import matplotlib.pyplot as plt
import json, os, time, shutil, sys, copy
import optuna
import torch
import sqlite3


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
        self.trial_n = 1
        self.normalized_hdataset = None

        self.synteticdataset_settings = get_datasets_config("graph_reasoning")
        self.graph_reasoning_settings_base = get_reasoning_config(f"same_{self.target_concept}_training")
        self.prepare_report_folder()
        self.prepare_dataset()
        self.set_hyperparameters_mappings()
        self.gnn_wrappers = {}


    def prepare_report_folder(self):
        self.report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"reports","synthetic", "training", self.graph_reasoning_settings_base["report"]["name"])
        self.summary_path = os.path.join(self.report_path, "summary")
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
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        
        combined_settings = {"dataset": self.synteticdataset_settings, "graph_reasoning": self.graph_reasoning_settings_base}
        with open(os.path.join(self.summary_path, "settings.json"), "w") as fp:
            json.dump(combined_settings, fp)

        self.db_path = f"{self.summary_path}/optuna_study.db"
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            conn.close()


    def prepare_dataset(self):
        dataset_generator = SyntheticDatasetGenerator(self.synteticdataset_settings, None, self.report_path)
        dataset_generator.create_dataset()
        settings_hdata = self.graph_reasoning_settings_base["hdata"]
        filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["noise"]
        extended_nxdatset = dataset_generator.extend_nxdataset(filtered_nxdataset, "training", "training")
        self.normalized_nxdatset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)
        gnn_wrapper = GNNWrapper(self.graph_reasoning_settings_base, "")
        gnn_wrapper.define_GCN()
        gnn_wrapper.set_nxdataset(self.normalized_nxdatset, None)
        self.normalized_hdataset = gnn_wrapper.hdataset

    def prepare_gnn(self, trial_n, report_path, graph_reasoning_settings):
        if not os.path.exists(report_path):
            os.makedirs(report_path)
        with open(os.path.join(report_path, "gnn_settings.json"), "w") as fp:
            json.dump(graph_reasoning_settings, fp)
        gnn_wrapper = GNNWrapper(graph_reasoning_settings, report_path, ID=trial_n)
        gnn_wrapper.define_GCN()
        gnn_wrapper.set_nxdataset(self.normalized_nxdatset, self.normalized_hdataset)
        return gnn_wrapper

    def set_hyperparameters_mappings(self):
        d = {}
        # d.update({"decoder_hc" : ["gnn", "decoder", "hidden_channels"]})
        d.update({"lr" : ["gnn", "lr"]})
        d.update({"enc_nod_hc" : ["gnn", "encoder", "nodes", "hidden_channels"]})
        d.update({"enc_edg_hc" : ["gnn", "encoder", "edges", "hidden_channels"]})
        d.update({"dec_hc_0" : ["gnn", "decoder", "hidden_channels", 0]})
        d.update({"dec_hc_1" : ["gnn", "decoder", "hidden_channels", 1]})
        # d.update({"dec_hc_2" : ["gnn", "decoder", "hidden_channels", 2]})
        self.hyperparameters_mappings = d

    def objective(self, trial):
        # Suggest hyperparameters to optimize
        hyperparameters_values = {}
        hyperparameters_values['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        # hyperparameters_values['num_layers'] = trial.suggest_int('num_layers', 1, 5)
        hyperparameters_values['enc_nod_hc'] = trial.suggest_int('enc_nod_hc', 8, 32)
        hyperparameters_values['enc_edg_hc'] = trial.suggest_int('enc_edg_hc', 8, 32)
        hyperparameters_values['dec_hc_0'] = trial.suggest_int('dec_hc_0', 64, 256)
        hyperparameters_values['dec_hc_1'] = trial.suggest_int('dec_hc_1', 32, 128)
        # hyperparameters_values['dec_hc_2'] = trial.suggest_int('dec_hc_2', 8, 256)

        print(f"Starting trial {trial.number} and parameters: {hyperparameters_values}")

        graph_reasoning_settings = self.update_settings_dict(self.graph_reasoning_settings_base, self.hyperparameters_mappings, hyperparameters_values)
        report_path = os.path.join(self.report_path, str(trial.number))
        gnn_wrapper = self.prepare_gnn(trial.number, report_path, graph_reasoning_settings)
        score = gnn_wrapper.train(verbose= True)
        gnn_wrapper.metric_subplot.close()
        self.gnn_wrappers[trial.number] = copy.deepcopy(gnn_wrapper)
        trial.set_user_attr("settings", copy.deepcopy(graph_reasoning_settings))

        plot_parallel_coordinate = optuna.visualization.plot_parallel_coordinate(self.study)
        plot_parallel_coordinate.write_image(os.path.join(self.report_path, f"parallel_coordinates_plot.png"))
        plot_optimization_history = optuna.visualization.plot_optimization_history(self.study)
        plot_optimization_history.write_image(os.path.join(self.report_path, f"plot_optimization_history.png"))
        plot_contour = optuna.visualization.plot_contour(self.study, params=['lr', 'enc_nod_hc'])  # Replace with relevant hyperparameters
        plot_contour.write_image(os.path.join(self.report_path, f"plot_contour.png"))
        
        return -score

    def hyperparameters_optimization(self):
        storage_path = f"sqlite:///{self.db_path}"
        max_retries = 5
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                self.study = optuna.create_study(
                    study_name=f"optimization_{self.target_concept}",
                    storage=storage_path,
                    direction="maximize",
                    load_if_exists=True
                )
                self.study.optimize(self.objective, n_trials=50, n_jobs=4)
                best_model = self.gnn_wrappers[self.study.best_trial.number]
                best_graph_reasoning_settings = self.study.best_trial.user_attrs["settings"]
                with open(os.path.join(self.summary_path, f"same_{self.target_concept}_best_optimization.json"), "w") as fp:
                    json.dump(best_graph_reasoning_settings, fp)
                best_model.save_model(os.path.join(self.summary_path, f"model_{self.target_concept}_best_optimization.pth"))
                best_model.metric_subplot.save(os.path.join(self.summary_path, f"model_{self.target_concept}_metric_subplot.png"))

                break

            except optuna.exceptions.StorageInternalError as e:
                if "database is locked" in str(e):
                    print(f"Attempt {attempt + 1}/{max_retries} failed with database lock. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise

    def standalone_train(self):
        self.graph_reasoning_settings = self.graph_reasoning_settings_base
        report_path = os.path.join(self.report_path, str("standalone"))
        gnn_wrapper = self.prepare_gnn("standalone", report_path, self.graph_reasoning_settings)
        score = gnn_wrapper.train(verbose= True)
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
# gnn_trainer.hyperparameters_optimization()
gnn_trainer.standalone_train()
# plt.show()
# input("press key")