from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from graph_visualizer import visualize_nxgraph
from GNNWrapper import GNNWrapper
import matplotlib.pyplot as plt
import json, os, time

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"config","SyntheticDataset", "graph_reasoning.json")) as f:
    synteticdataset_settings = json.load(f)
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"config","GraphReasoning", "same_room_training.json")) as f:
    graph_reasoning_settings = json.load(f)

dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings)
# # room_clustering_dataset = dataset_generator.get_ws2room_clustering_datalodaer()
filtered_nxdataset = dataset_generator.get_filtered_datset(["ws"],["ws_same_room"])["original"]
extended_nxdatset = dataset_generator.extend_nxdataset(filtered_nxdataset)
# visualize_nxgraph(extended_nxdatset["val"][0], "graph_matching inital graph")
# plt.show()
# time.sleep(20)
# # # dataset_generator.reintroduce_predicted_edges(new_nxdatset["train"][0], [], "testing custom mp graph")

gnn_wrapper = GNNWrapper(extended_nxdatset["train"], graph_reasoning_settings)

gnn_wrapper.define_GCN()

gnn_wrapper.train(verbose= True)

# # gt_base_graphnx, unparented_base_graphnx, hdata_graph, node_label_mapping, ground_truth, gt_edges = dataset.get_ws2room_clustering_single_base_knn_graph(visualize=True)
predicted_edges = gnn_wrapper.infer(extended_nxdatset["inference"], True)
# inference_base_graph = new_nxdatset["inference"][0]
# dataset_generator.reintroduce_predicted_edges(inference_base_graph, [], "Inference: ground truth")
# inference_base_graph.remove_all_edges()
# dataset_generator.reintroduce_predicted_edges(inference_base_graph, predicted_edges, "Inference: predictions")

# mp_edges, label_edges = gnn_wrapper.get_message_sharing_edges(gt_base_graphnx)
# dataset.reintroduce_predicted_edges(unparented_base_graphnx, mp_edges["train"], "train: mp_edges")
# dataset.reintroduce_predicted_edges(unparented_base_graphnx, label_edges["train"], "train: label_edges")
# dataset.reintroduce_predicted_edges(unparented_base_graphnx, mp_edges["val"], "val: mp_edges")
# dataset.reintroduce_predicted_edges(unparented_base_graphnx, label_edges["val"], "val: label_edges")
# dataset.reintroduce_predicted_edges(unparented_base_graphnx, mp_edges["test"], "test: mp_edges")
# dataset.reintroduce_predicted_edges(unparented_base_graphnx, label_edges["test"], "test: label_edges")

plt.show()