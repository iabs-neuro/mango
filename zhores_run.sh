#!/bin/bash -l


# ------------
# --- Options:

#SBATCH --job-name=a.chertkov_ntt
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=0-40:00:00
#SBATCH --partition gpu
##SBATCH --mem-per-cpu=1500MB
##SBATCH --mem=5GB
#SBATCH --mail-type=ALL
#SBATCH --output=zhores_out.txt


# ------------------------------------
# --- Install manually before the run:

# module load python/anaconda3
# conda activate && conda remove --name neural_tensor_train --all -y
# conda create --name neural_tensor_train python=3.8 -y
# conda activate neural_tensor_train
# pip install "jax[cpu]==0.4.3" optax teneva==0.13.2 ttopt==0.5.0 protes==0.2.3 torch torchvision snntorch scikit-image matplotlib nevergrad requests urllib3


# ----------------
# --- Main script:
module rm *
module load python/anaconda3
module load gpu/cuda-12.0
conda activate neural_tensor_train

# srun python3 manager.py --data cifar10 --task check --kind data
# srun python3 manager.py --data cifar10 --model densenet --task check --kind model --c 0
# srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task train --kind gen
# srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task check --kind gen
# srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task check --kind gen
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 0
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 0
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 1
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 1
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 2
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 2
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 3
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 3
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 4
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 4
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 5
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 5
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 6
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 6
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 7
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 7
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 8
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 8
srun python3 manager.py --data cifar10 --gen vae_vq --model densenet --task am --kind class --c 9
srun python3 manager.py --data cifar10 --gen gan_sn --model densenet --task am --kind class --c 9

exit 0


# ---------------------------------
# --- How to use this shell script:
# Run this script as "sbatch zhores_run.sh"
# Check status as: "squeue"
# Delete the task as "scancel NUMBER"
