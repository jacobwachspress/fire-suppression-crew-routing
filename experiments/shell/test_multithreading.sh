#!/bin/bash

module load julia/1.10.1
module load gurobi/gurobi-1102


julia --threads 32 --project=package_dependencies/julia test/test_multithreading.jl