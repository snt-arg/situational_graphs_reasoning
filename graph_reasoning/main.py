from CustomSquaredRoomDataset import CustomSquaredRoomNetworkxGraphs
from GNNWrapper import GNNWrapper
import matplotlib.pyplot as plt
import json, os

# grid_dims = [5,5]
# room_center_distances = [5,5]
# wall_thickness = 0.5
# max_room_entry_size = 1
# n_buildings = 1000

# val_ratio = 0.1
# test_ratio = 0.1

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"config","config.json")) as f:
    settings = json.load(f)

dataset = CustomSquaredRoomNetworkxGraphs(settings)
room_clustering_dataset = dataset.get_ws2room_clustering_datalodaer()

gnn_wrapper = GNNWrapper(room_clustering_dataset, settings)

gnn_wrapper.define_GCN()

# gnn_wrapper.train(verbose= True)

gt_base_graphnx, unparented_base_graphnx, hdata_graph, node_label_mapping, ground_truth, gt_edges = dataset.get_ws2room_clustering_single_base_knn_graph(visualize=True)
# preds = gnn_wrapper.infer(hdata_graph, ground_truth)
# dataset.reintroduce_predicted_edges(unparented_base_graphnx, preds, "Inference: predictions")
mp_edges, label_edges = gnn_wrapper.get_message_sharing_edges(gt_base_graphnx)

dataset.reintroduce_predicted_edges(unparented_base_graphnx, mp_edges["train"], "mp_edges")
dataset.reintroduce_predicted_edges(unparented_base_graphnx, label_edges["train"], "label_edges")

plt.show()