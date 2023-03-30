#!/bin/bash -l

#SBATCH --job-name=a.chertkov_ntt
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --time=0-20:00:00
#SBATCH --partition gpu
##SBATCH --mem-per-cpu=1500MB
##SBATCH --mem=5GB
#SBATCH --mail-type=ALL
#SBATCH --output=out_zhores.txt

# --- Install before the run
# module load python/anaconda3
# conda activate && conda remove --name neural_tensor_train --all -y
# conda create --name neural_tensor_train python=3.8 -y
# conda activate neural_tensor_train
# pip install "jax[cpu]==0.4.3" optax teneva==0.13.2 ttopt==0.5.0 protes==0.2.3 torch torchvision snntorch scikit-image matplotlib PyYAML nevergrad requests urllib3


# --- Main script
module rm *
module load python/anaconda3
module load gpu/cuda-12.0
conda activate neural_tensor_train
srun python3 manager.py gan_sn_check gan_sn_inv

exit 0


# --- How to use this shell script
# Run this script as "sbatch run_zhores.sh"
# Check status as: "squeue"
# Delete the task as "scancel NUMBER"
