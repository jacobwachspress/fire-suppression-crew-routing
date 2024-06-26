#!/bin/bash

#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

module load julia/1.10.1
module load gurobi/gurobi-1102


julia --threads 32 --project=package_dependencies/julia experiments/profile.jl