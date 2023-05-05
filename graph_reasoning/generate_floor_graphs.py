import numpy as np
import networkx as nx
from graph_visualizer import visualize_nxgraph
import copy



def generate_floor_graphs(grid_dims, room_center_distances, wall_thickness, max_room_entry_size, floors_number):

    floor_graphs = []
    for floor_n in range(floors_number):
        graph = nx.Graph()

        ### Base matrix
        base_matrix = np.zeros(grid_dims)
        room_n = 0
        for i in range(base_matrix.shape[0]):
            for j in range(base_matrix.shape[1]):
                if base_matrix[i,j] == 0.:
                    room_n += 1
                    aux_col = np.where(base_matrix[i:,j] != 0)[0]
                    aux_row = np.where(base_matrix[i,j:] != 0)[0]
                    if len(aux_col) != 0:
                        remaining_x = aux_col[0]
                    else:
                        remaining_x = len(base_matrix[i:,j])
                    if len(aux_row) != 0:
                        remaining_y = aux_row[0]
                    else:
                        remaining_y = len(base_matrix[i,j:])
                    remaining = [remaining_x, remaining_y]
                    room_entry_size = [min(remaining[0], np.random.randint(low=1, high=max_room_entry_size+1, size=(1))[0]),\
                                    min(remaining[1], np.random.randint(low=1, high=max_room_entry_size+1, size=(1))[0])]
                    for ii in range(room_entry_size[0]):
                        for jj in range(room_entry_size[1]):
                            base_matrix[i+ii, j+jj] = room_n

                    node_ID = len(graph.nodes)
                    room_center = [room_center_distances[0]*(i + ii/2), room_center_distances[1]*(j+jj/2)]
                    room_area = [room_center_distances[0]*room_entry_size[0] - wall_thickness*2, room_center_distances[1]*room_entry_size[1] - wall_thickness*2]
                    graph.add_nodes_from([(node_ID,{"center" : room_center, "room_area" : room_area, "viz_type" : "Point", "viz_data" : room_center, "viz_feat" : 'bx'})])
                    

        total_rooms_n = room_n

        ### Wall surfaces
        nodes_data = copy.deepcopy(graph.nodes(data=True))
        for node_data in nodes_data:
            normals = [[1,0],[-1,0],[0,1],[0,-1]]
            for i in range(4):
                node_ID = len(graph.nodes)
                orthogonal_normal = np.rot90([normals[i]]).reshape(2)
                ws_center = node_data[1]["center"] + np.array(normals[i])*np.array(node_data[1]["room_area"])/2
                ws_limit_1 = ws_center + np.array(orthogonal_normal)*np.array(node_data[1]["room_area"])/2
                ws_limit_2 = ws_center + np.array(-orthogonal_normal)*np.array(node_data[1]["room_area"])/2
                graph.add_nodes_from([(node_ID,{"center" : ws_center, "viz_type" : "Line", "viz_data" : [ws_limit_1,ws_limit_2], "viz_feat" : 'k'})])
                graph.add_edges_from([(node_ID,node_data[0])])

        visualize_nxgraph(graph)

        # ### Walls
        # wall_base_matrix = np.zeros(grid_dims*np.array([2,2]) - np.array([1,1]))

        # print("wall_base_matrix {}".format(wall_base_matrix.shape))
        floor_graphs.append(graph)


    return floor_graphs