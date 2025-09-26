#!/bin/bash
set -e

# setup ros2 environment

cp /root/.gitconfig_local /root/.gitconfig
git config --global --add safe.directory '*'

exec "python src/situational_graphs_reasoning/src/graph_reasoning/graph_reasoning_node.py"
