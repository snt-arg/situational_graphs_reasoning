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
view1 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 1})
view2 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 2})
view3 = dataset_generator.graphs["views"][0].filter_graph_by_node_attributes_containted({"view" : 3})
visualize_nxgraph(dataset_generator.graphs["original"][0], "original")
visualize_nxgraph(dataset_generator.graphs["noise"][0], "noise")
# visualize_nxgraph(view1, "with views 1")
# visualize_nxgraph(view2, "with views 2")
# visualize_nxgraph(view3, "with views 3")

# visualize_nxgraph(dataset_generator.get_filtered_datset(["ws"],["ws_same_room"])[0], "train data")
plt.show()