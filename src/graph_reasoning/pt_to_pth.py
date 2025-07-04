import torch

scripted = torch.jit.load("/home/adminpc/workspaces/reasoning_ws/src/graph_factor_nn/torchscripts/room_msd.pt", map_location="cpu")  # <- the .pt
state_dict = scripted.state_dict()                            # OrderedDict of tensors
torch.save(state_dict, "/home/adminpc/workspaces/reasoning_ws/src/graph_factor_nn/pths/room_msd.pth")  