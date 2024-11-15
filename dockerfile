FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Keys
ARG ssh_prv_key
ARG ssh_pub_key

# Work keys
RUN  apt-get -yq update && apt-get -yqq install ssh
RUN mkdir -p -m 0700 /root/.ssh && \
ssh-keyscan -H github.com >> /root/.ssh/known_hosts

RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
chmod 600 /root/.ssh/id_rsa && \
chmod 600 /root/.ssh/id_rsa.pub

RUN apt-get -y install python3-pip

# Install git
RUN apt-get update && apt-get install -y git

RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get install -y wget bzip2 ca-certificates curl git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# # Install essential packages
# RUN apt-get -yqq update && \
#     apt-get install -yq --no-install-recommends \
#     software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -yq bzip2 ca-certificates curl git wget ssh && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Pip install deps
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir wandb
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir colorama
RUN pip install --no-cache-dir seaborn
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir optuna
RUN pip install --no-cache-dir yacs tensorboard
RUN pip install --no-cache-dir networkx==3.1
RUN pip install --no-cache-dir plotly

# Install torch and deps
RUN pip install --no-cache-dir torch==2.2.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir torch-geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cu118.html

# Install graph wrapper and deps
WORKDIR /root/workspaces/reasoning_ws
RUN git clone -b feat/reasoning git@github.com:snt-arg/situational_graphs_wrapper.git
RUN mv /root/workspaces/reasoning_ws/situational_graphs_wrapper /root/workspaces/reasoning_ws/graph_wrapper
WORKDIR /root/workspaces/reasoning_ws/graph_wrapper
RUN pip install .

# Install graph matching and deps
WORKDIR /root/workspaces/reasoning_ws
RUN git clone -b feat/params_grid_search git@github.com:snt-arg/graph_matching.git
WORKDIR /root/workspaces/reasoning_ws/graph_matching
RUN pip install .
RUN pip install transforms3d

# Install graph reasoning and deps
WORKDIR /root/workspaces/reasoning_ws
RUN git clone -b train/bigger_rooms git@github.com:snt-arg/situational_graphs_reasoning.git
RUN mv /root/workspaces/reasoning_ws/situational_graphs_reasoning /root/workspaces/reasoning_ws/graph_reasoning
WORKDIR /root/workspaces/reasoning_ws/graph_reasoning
RUN pip install .

# Install graph datasets and deps
WORKDIR /root/workspaces/reasoning_ws
RUN git clone -b graph_reasoning git@github.com:snt-arg/situational_graphs_datasets.git
RUN mv /root/workspaces/reasoning_ws/situational_graphs_datasets /root/workspaces/reasoning_ws/graph_datasets
WORKDIR /root/workspaces/reasoning_ws/graph_datasets
RUN pip install .
# RUN pip install transforms3d

# Install graph datasets and deps
WORKDIR /root/workspaces/reasoning_ws
RUN git clone -b feature/3ws_rooms git@github.com:snt-arg/graph_factor_nn.git
WORKDIR /root/workspaces/reasoning_ws/graph_factor_nn
RUN pip install .

# Removing SSH Host authorization (GitHub)
RUN rm -rf /root/.ssh/

WORKDIR /root/workspaces/reasoning_ws

# RUN apt-get install nano
RUN echo 'alias re="python3 graph_reasoning/src/graph_reasoning/synthdata_training_stack.py"' >> ~/.bashrc

# Shell commands in Remote machine
# python main.py device 0 dataset scene_graphs task local_denoising diffusion.num_steps 20

#### Shell commands in Local machine
## docker build --build-arg ssh_prv_key="$(cat ~/.ssh/id_ed25519)" --build-arg ssh_pub_key="$(cat ~/.ssh/id_ed25519.pub)"  -t reasoning:original .
### docker save reasoning:original -o reasoning.tar
### singularity -d build reasoning.sif docker-archive://reasoning.tar
## singularity -d build reasoning.sif docker-daemon://reasoning:original
## rsync --rsh='ssh -p 8022' -avzu reasoning.sif  jmillan@access-iris.uni.lu:workspace
## ssh iris-cluster

#### Shell command in iris machine
## si
## lsi
## rsi
## sjob
## sacct