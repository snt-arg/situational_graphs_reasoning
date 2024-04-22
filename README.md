# Graph Reasoning

## Overview

Graph Reasoning extract high-level entities from lower layers of S-Graphs.

### License

The source code is released under GPLv3 License [![License: GPLv2](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://opensource.org/licenses/MIT).

**Author: Jose Andres Millan Romera<br />
Affiliation: [University of Luxembourg](https://www.anybotics.com/)<br />
Maintainer: Jose Andres Millan Romera, josmilrom@gmail.com**

The graph_matching package has been tested under [ROS2] Humble on Ubuntu 20.04.
This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.


## Installation

### Installation from Packages

The only tested ROS version for this package is ROS2 Humble

## Usage

Run the graph_reasoning node with

	ros2 launch graph_reasoning graph_reasoning.launch.py 

## Config files

config/

* **same_room_training.json** Describes the data preprocessing and the GNN hyperparameters for room generation. 

* **same_wall_training.json** Describes the data preprocessing and the GNN hyperparameters for wall generation. 

## Launch files

* **graph_reasoning.launch.py  :** Launch of graph_reasoning node


## Nodes

#### Subscribed Topics

* **`/s_graphs/all_map_planes`** ([s_graphs/PlanesData]) Used for room entities generation
* **`/s_graphs/map_planes`** ([s_graphs/PlanesData]) Used for wall entities generation

#### Published Topics

* **`/room_segmentation/room_data`** ([s_graphs/RoomsData])
* **`/room_segmentation/wall_data`** ([s_graphs/WallsData])

#### Parameters (TODO)

* **`subscriber_topic`** (string, default: "/temperature")

	The name of the input topic.

* **`cache_size`** (int, default: 200, min: 0, max: 1000)

	The size of the cache.




