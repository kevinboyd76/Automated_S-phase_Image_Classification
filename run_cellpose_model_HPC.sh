#!/bin/bash -l

module purge
module load slurm python/3.10.2 cellpose scikit-learn

my_current_dir=$( dirname $0)

if [ "$1" == "--help" ]; then
    $my_current_dir/run_cellpose_model.py --help
    exit
fi

sbatch $my_current_dir/run_cellpose_model.py $@ 
