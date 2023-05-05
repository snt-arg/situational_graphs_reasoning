from generate_floor_graphs import generate_floor_graphs
from train_room_clustering import train_room_clustering

grid_dims = [10,10]
room_center_distances = [5,5]
wall_thickness = 0.5
max_room_entry_size = 5
floors_number = 2


graphs = generate_floor_graphs(grid_dims, room_center_distances, wall_thickness, max_room_entry_size, floors_number)
train_room_clustering(graphs)