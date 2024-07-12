import os, sys
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import ZINC
import torch

graph_reasoning_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),"graph_reasoning")
sys.path.append(graph_reasoning_dir)
from graph_reasoning.from_networkxwrapper_2_heterodata import from_networkxwrapper_2_heterodata
graph_wrapper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"graph_wrapper")
sys.path.append(graph_wrapper_dir)
from graph_wrapper.GraphWrapper import GraphWrapper

from grapharm import GraphARM
from models import DiffusionOrderingNetwork, DenoisingNetwork
from utils import NodeMasking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

### FAKE DATASET
nodes_def = [(0, {"type" : "0", "x" : [0]}), (1, {"type" : "0", "x" : [1]}), (2, {"type" : "0", "x" : [2]})]
edges_def = [(0, 1, {"type" : "0", "x" : [0]}), (1, 2, {"type" : "0", "x" : [1]}), (0, 2, {"type" : "0", "x" : [2]})]
graph_def = {"nodes" : nodes_def, "edges": edges_def, "name": "test"}

graph = GraphWrapper(graph_def)
hdata = from_networkxwrapper_2_heterodata(graph)

dataset = Data(x=hdata['0'].x, edge_index=hdata["0","0","0"].edge_index, edge_attr=hdata["0","0","0"].x)
# dataset = ZINC(root='~/workspace/GraphDiffusionImitate/data/ZINC', transform=None, pre_transform=None)
print(f"flag dataset {dataset}")
### NETWORKS
diff_ord_net = DiffusionOrderingNetwork(node_feature_dim=1,
                                        num_node_types=dataset.x.unique().shape[0],
                                        num_edge_types=dataset.edge_attr.unique().shape[0],
                                        num_layers=3,
                                        out_channels=1,
                                        device=device)

# masker = NodeMasking(dataset)


denoising_net = DenoisingNetwork(
    node_feature_dim=dataset.num_features,
    edge_feature_dim=dataset.num_edge_features,
    num_node_types=dataset.x.unique().shape[0],
    num_edge_types=dataset.edge_attr.unique().shape[0],
    num_layers=7,
    # hidden_dim=32,
    device=device
)

torch.autograd.set_detect_anomaly(True)

grapharm = GraphARM(
    dataset=dataset,
    denoising_network=denoising_net,
    diffusion_ordering_network=diff_ord_net,
    device=device
)

dataset = [dataset,dataset,dataset]
batch_size = 1
try:
    grapharm.load_model()
    print("Loaded model")
except:
    print ("No model to load")
# train loop
for epoch in range(1):
    print(f"Epoch {epoch}")
    grapharm.train_step(
        train_data=dataset[2*epoch*batch_size:(2*epoch + 1)*batch_size],
        val_data=dataset[(2*epoch + 1)*batch_size:batch_size*(2*epoch + 2)],
        M=4
    )
    grapharm.save_model()