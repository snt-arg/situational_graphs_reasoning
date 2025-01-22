import matplotlib.pyplot as plt
import json, os, time, shutil, sys, copy, datetime
import optuna
import torch
import sqlite3
import numpy as np

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
        self.normalized_hdataset = None

        self.synteticdataset_settings = get_datasets_config("graph_reasoning")
        self.graph_reasoning_settings_base = get_reasoning_config(f"same_{self.target_concept}_training")
        self.gnn_wrappers = {}


    def prepare_report_folder(self, resuming):
        now_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"reports","synthetic", "training", self.graph_reasoning_settings_base["report"]["name"], now_timestamp)
        self.summary_path = os.path.join(self.report_path, "summary")
        if not os.path.exists(self.report_path):
            os.makedirs(self.report_path)
        elif not resuming:
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
        
        combined_settings = {"dataset": self.synteticdataset_settings, "graph_reasoning": self.graph_reasoning_settings_base, "time":now_timestamp}
        with open(os.path.join(self.summary_path, "settings.json"), "w") as fp:
            json.dump(combined_settings, fp)

        self.db_path = f"{self.summary_path}/optuna_study.db"
        db_available = False
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            conn.close()
            db_available = True
        elif resuming:
            try:
                optuna.create_study(study_name="", storage=self.db_path)
                shutil.copy(self.db_path, f"{self.summary_path}/optuna_study_backup.db")
                db_available = True
            except:
                print(f"Bayesian training HP optimization: The current database file is corrupted. Path: {self.db_path}")
                db_available = False
        elif not resuming:
            os.unlink(self.db_path)
            conn = sqlite3.connect(self.db_path)
            conn.close()
            db_available = True
        return db_available
            

    def prepare_dataset(self):
        dataset_generator = SyntheticDatasetGenerator(self.synteticdataset_settings, None, self.report_path)
        dataset_generator.create_dataset()
        # settings_hdata = self.graph_reasoning_settings_base["hdata"]
        # filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["noise"]
        extended_nxdatset = dataset_generator.extend_nxdataset(dataset_generator.graphs["noise"], "training", "training")
        self.normalized_nxdatset = dataset_generator.normalize_features_nxdatset(extended_nxdatset)
        # dataset_generator.dataset_to_hdata()
        # asdf
        gnn_wrapper = GNNWrapper(self.graph_reasoning_settings_base, self.summary_path)
        gnn_wrapper.define_GCN()
        gnn_wrapper.set_nxdataset(self.normalized_nxdatset, None)
        self.normalized_hdataset = gnn_wrapper.hdataset

        torch.save(self.normalized_hdataset, 'hetero_data_list.pt')
        gnn_wrapper.visualize_hetero_features()

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
        for name, path_list in self.graph_reasoning_settings_base["hyperp_bay_optim"]["mapping"].items():
            d.update({name : path_list})
        self.hyperparameters_mappings = d

    def objective(self, trial):
        hyperparameters_values = {}
        for name, attrs in self.graph_reasoning_settings_base["hyperp_bay_optim"]["range"].items():

            if attrs["num_type"] == "float":
                hyperparameters_values[name] = trial.suggest_float(name, attrs["limits"][0], attrs["limits"][1])
            elif attrs["num_type"] == "int":
                hyperparameters_values[name] = trial.suggest_int(name, attrs["limits"][0], attrs["limits"][1])
            elif attrs["num_type"] == "loguniform":
                hyperparameters_values[name] = trial.suggest_loguniform(name, attrs["limits"][0], attrs["limits"][1])
            
        print(f"Starting trial {trial.number} with parameters: {hyperparameters_values}")

        graph_reasoning_settings = self.update_settings_dict(self.graph_reasoning_settings_base, self.hyperparameters_mappings, hyperparameters_values)
        report_path = os.path.join(self.report_path, str(trial.number))
        gnn_wrapper = self.prepare_gnn(trial.number, report_path, graph_reasoning_settings)
        score = gnn_wrapper.train(verbose= True)
        gnn_wrapper.metric_subplot.close()
        gnn_wrapper.free_gpu_memory()
        self.gnn_wrappers[trial.number] = gnn_wrapper
        trial.set_user_attr("settings", copy.deepcopy(graph_reasoning_settings))

        plot_parallel_coordinate = optuna.visualization.plot_parallel_coordinate(self.study)
        # objective_values = [trial.value for trial in self.study.trials if trial.value is not None]

        # # Check if objective_values is empty
        # if not objective_values:
        #     print("No objective values found in the trials. Skipping color scale adjustment.")
        # else:
        #     # Normalize values for color mapping
        #     normalized_values = [(val - min(objective_values)) / (max(objective_values) - min(objective_values)) 
        #                         if max(objective_values) != min(objective_values) else 1 
        #                         for val in objective_values]

        #     # Update the line color in the parallel coordinates plot
        #     plot_parallel_coordinate.update_traces(
        #         line=dict(
        #             color=objective_values,  # Use original objective values for color mapping
        #             colorscale="Viridis",   # Choose a color scale
        #             showscale=True,         # Display the color bar
        #             cmin=min(objective_values),
        #             cmax=max(objective_values)
        #         )
        #     )

        # # Update layout for aesthetics
        # plot_parallel_coordinate.update_layout(
        #     coloraxis_colorbar=dict(
        #         title="Objective Value",
        #         ticksuffix="",
        #         showticksuffix="last"
        #     )
        # )
        plot_parallel_coordinate.write_image(os.path.join(self.summary_path, f"parallel_coordinates_plot.png"))
        plot_optimization_history = optuna.visualization.plot_optimization_history(self.study)
        plot_optimization_history.write_image(os.path.join(self.summary_path, f"plot_optimization_history.png"))
        # plot_contour = optuna.visualization.plot_contour(self.study, params=['lr', 'enc_nod_hc'])  # Replace with relevant hyperparameters
        # plot_contour.write_image(os.path.join(self.summary_path, f"plot_contour.png"))
        # score = np.expm1(score)
        return -score

    def hyperparameters_optimization(self):
        self.prepare_report_folder(self.graph_reasoning_settings_base["hyperp_bay_optim"]["resume"])
        self.prepare_dataset()
        self.set_hyperparameters_mappings()
        storage_path = f"sqlite:///{self.db_path}"
        max_retries = 5
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                self.study = optuna.create_study(
                    study_name=f"optimization_{self.target_concept}",
                    storage=storage_path,
                    direction="maximize",
                    load_if_exists=self.graph_reasoning_settings_base["hyperp_bay_optim"]["resume"], 
                )
                if self.graph_reasoning_settings_base["hyperp_bay_optim"]["use_init_values"]:
                    hp_initial_values_dict = self.get_initial_hp_values(self.graph_reasoning_settings_base, self.hyperparameters_mappings)
                    self.study.enqueue_trial(hp_initial_values_dict)

                if self.graph_reasoning_settings_base["hyperp_bay_optim"]["dynamic_n_jobs"]:
                    num_gpus = torch.cuda.device_count()
                    n_jobs = num_gpus if num_gpus > 0 else 1
                else:
                    n_jobs = 1
                    
                self.study.optimize(self.objective, n_trials=self.graph_reasoning_settings_base["hyperp_bay_optim"]["n_trials"], n_jobs=n_jobs)
                if self.study.best_trial.number in self.gnn_wrappers.keys():
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
        self.prepare_report_folder(False)
        self.prepare_dataset()
        self.set_hyperparameters_mappings()
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
    
    def get_initial_hp_values(self, base_settings, mappings):
        def get_nested_value(nested_dict, keys):
            current = nested_dict
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
                    current = current[key]
                else:
                    return None
            return current
        
        values_dict = {}
        for hp_name in mappings.keys():
            values_dict[hp_name] = get_nested_value(base_settings, mappings[hp_name])

        return values_dict
    

gnn_trainer = GNNTrainer()
gnn_trainer.hyperparameters_optimization()
# gnn_trainer.standalone_train()
# plt.show()
# input("press key")

# optuna-dashboard sqlite:////home/adminpc/workspaces/reasoning_ws/src/situational_graphs_reasoning/src/reports/synthetic/training/RoomWall/summary/optuna_study.db