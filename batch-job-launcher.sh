#!/bin/bash -l
#SBATCH -c 14
#SBATCH --time=0-06:00:00
#SBATCH -p gpu
#SBATCH -G 2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jose.millan@uni.lu

module load tools/Singularity
singularity exec --nv ~/workspace/reasoning_ws/situational_graphs_reasoning/reasoning.sif python3 ~/workspace/reasoning_ws/situational_graphs_reasoning/src/graph_reasoning/synthdata_training_stack.py

