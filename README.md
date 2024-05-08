<div align="center">
 <h1>Situational Graphs - Reasoning</h1>
</div>

**Situational Graphs - Reasoning** is a ROS2 package for generating in real-time semantic concepts like **_Rooms_** and **_Walls_** from **_Wall Surfaces_** [S-Graphs](https://uni.lu/)<br />. For that purpose, Graph Neural Networks (GNNs) are used to estimate the existing relations between the wall surfaces. 

## 📜 Table of contents

- [📖 Published Papers](#published-papers)
- [⚙️ Installation](#installation)
  - [📦 Installation with S-Graphs](#installation-with-sgraphs)
  - [📦 Installation From Source](#installation-from-source)
- [🚀 Usage](#usage)
- [⚙️ Configuration files](#config-files)
- [🤖 ROS Related](#ros-related)
  - [📥 Subscribed Topics](#subscribed-topics)
  - [📤 Published Topics](#published-topics)
  - [🔄 ROS Services](#ros-services)
  - [⚙️ ROS Parameters](#ros-parameters)
  - [🌐 Published TFs](#published-tfs)

## 📖 Published Papers <a id="published-papers"></a>

<details >
    <summary><a href="https://arxiv.org/abs/2310.00401">Learning High-level Semantic-Relational Concepts for SLAM </a>
    </summary>

</details>

## ⚙️ Installation <a id="installation"></a>

<!-- TODO: When s-graphs is available in rosdistro add here the command to install -->

> [!NOTE]
> Situational Graphs - Reasoning was only tested on Ubuntu 20.04, ROS2 Foxy, Humble Distros.
> We strongly recommend using [cyclone_dds](https://docs.ros.org/en/humble/Installation/DDS-Implementations/Working-with-Eclipse-CycloneDDS.html) instead of the default fastdds.

### 📦 Installation with S-Graphs <a id="installation-with-sgraphs"></a>

Follow the [S-Graphs installation instructions](github.com:snt-arg/lidar_s_graphs.git)

### 📦 Installation From Source <a id="installation-from-source"></a>

> [!IMPORTANT]
> Before proceeding, make sure you have `rosdep` installed. You can install it using `sudo apt-get install python3-rosdep`
> In addition, ssh keys are needed to be configured on you GitHub account. If you haven't
> yet configured ssh keys, follow this [tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

1. Update Rosdep:

```sh
rosdep init && rosdep update --include-eol-distros
```

2. Create a ROS2 workspace for S-Graphs

```bash
mkdir -p $HOME/workspaces && cd $HOME/workspaces
```

3. Clone the S-Graphs repository into the created workspace

```bash
git clone git@github.com:snt-arg/situational_graphs_reasoning.git -b develop
```

> [!IMPORTANT]
> If you have Nvidia GPU please install CUDA from this [link](https://developer.nvidia.com/cuda-11-8-0-download-archive). This code has only been tested with CUDA 11.8.
> If you dont have CUDA S-Graphs will use CPU only.

4. Install required dependencies. Change $ROS_DISTRO to your ros2 version.

```bash
cd situational_graphs_reasoning && source /opt/ros/$ROS_DISTRO/setup.sh && pip3 install -r requirements.txt
```

> [!NOTE]
> If you want to compile with debug traces (from backward_cpp) run:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## 🚀 Usage <a id="usage"></a>

Follow the [S-Graphs instructions](github.com:snt-arg/lidar_s_graphs.git) to use this package along with all other functionalities.

Or launch situational_graphs_reasoning.py.

## ⚙️ Configuration files <a id="config-files"></a>

| File name                     | Description                                                                          |
| ------------------------------ | ----------------------------------------------------------------------------------- |
| `config/same_room_training.json` | Describes the data preprocessing and the GNN hyperparameters for room generation. |
| `config/same_wall_training.json` | Describes the data preprocessing and the GNN hyperparameters for wall generation. |

## 🤖 ROS Related <a id="ros-related"></a>

### 📥 Subscribed Topics <a id="subscribed-topics"></a>

#### `situational_graphs_reasoning_node` node

| Topic name         | Message Type                                                                                        | Description                              |
| -------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `/s_graphs/all_map_planes` | [s_graphs/PlanesData](https://github.com/snt-arg/s_graphs/blob/feature/ros2/msg/PlanesData.msg)              | Al the plains contained in the map. |
| `/s_graphs/map_planes`     | [s_graphs/PlanesData](https://github.com/snt-arg/s_graphs/blob/feature/ros2/msg/PlanesData.msg) | Only the plains incorporated with the last keypoint.    |

### 📤 Published Topics <a id="published-topics"></a>

#### `situational_graphs_reasoning_node` node

| Topic name                     | Message Type                                                                                  | Description                                                              |
| ------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `/room_segmentation/room_data` | [s_graphs/RoomsData](https://github.com/snt-arg/s_graphs/blob/feature/ros2/msg/RoomsData.msg) | Contains all the necessary information about the rooms on a given floor. |
| `/room_segmentation/wall_data` | [s_graphs/WallsData](https://github.com/snt-arg/s_graphs/blob/feature/ros2/msg/WallsData.msg) | Contains all the necessary information about the walls on a given floor. |

### 🔄 ROS Services <a id="ros-services"></a>

| Topic name       | Message Type                                                                                  | Description                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `/s_graphs/dump` | [s_graphs/DumpGraph](https://github.com/snt-arg/s_graphs/blob/feature/ros2/srv/DumpGraph.srv) | Save all the internal data (point clouds, floor coeffs, odoms, and pose graph) to a directory. |

| Topic name           | Message Type                                                                              | Description                              |
| -------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------- |
| `/s_graphs/save_map` | [s_graphs/SaveMap](https://github.com/snt-arg/s_graphs/blob/feature/ros2/srv/SaveMap.srv) | Save the generated 3D map as a PCD file. |

### ⚙️ ROS Parameters <a id="ros-parameters"></a>

All the configurable parameters are listed in config folder as ros params.

### 🌐 Published TFs <a id="published-tfs"></a>

- `map2odom`: The transform published between the map frame and the odom frame after the corrections have been applied.

- The entire `tf_tree` for the virtual experiment can be seen in the figure below.

<p align="center">
  <a href="">
    <img src="./imgs/Tf-tree.png" alt="tf_tree" width="80%">
  </a>
</p>









### License

The source code is released under GPLv3 License [![License: GPLv3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://opensource.org/license/gpl-3-0).

**Author: Jose Andres Millan Romera<br />
Affiliation: [University of Luxembourg](https://uni.lu/)<br />
Maintainer: Jose Andres Millan Romera, <josmilrom@gmail.com>**

The graph_matching package has been tested under [ROS2] Humble on Ubuntu 20.04.
This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.

## Installation

### Installation from Packages

The only tested ROS version for this package is ROS2 Humble

## Usage

Run the situational_graphs_reasoning node with

 ros2 launch situational_graphs_reasoning situational_graphs_reasoning.launch.py

## Config files

config/

* **same_room_training.json** Describes the data preprocessing and the GNN hyperparameters for room generation.

* **same_wall_training.json** Describes the data preprocessing and the GNN hyperparameters for wall generation.

## Launch files

* **situational_graphs_reasoning.launch.py  :** Launch of situational_graphs_reasoning node

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
