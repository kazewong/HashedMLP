#! /bin/bash

#SBATCH --job-name=hashed_neural_field_camels
#SBATCH --partition=gpu
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --constraint=a100|v100-32gb
#SBATCH --time=2:00:00

set -e

DATA_PATH="/mnt/ceph/users/wzhou/projects/fi-cosmology/data/compression/athena/rho_1024.npy"
OUTPUT_PATH="/mnt/ceph/users/${USER}/projects/fi-cosmology/outputs/compression/hashed_neural_field/athena/$SLURM_JOB_ID"
IMAGE="/mnt/ceph/users/wzhou/images/cosmology.sif"
NORMALIZATION=${NORMALIZATION:-linear}
HASHTABLE_SIZE_LOG2=${HASHTABLE_SIZE_LOG2:-24}
EPOCHS=${EPOCHS:-60}

module load singularity

singularity run --nv --containall --writable-tmpfs -B $HOME/.ssh -B $PWD -B /mnt/ceph/users --pwd $PWD $IMAGE \
    python -m finfc.train hydra.run.dir="${OUTPUT_PATH}/train" \
        data.path=${DATA_PATH} data.normalize=$NORMALIZATION data.num_workers=4 \
        model.hash.num_entries_log2=${HASHTABLE_SIZE_LOG2} \
        batch_size=$((2 ** 22)) max_epochs=${EPOCHS}

singularity run --nv --containall --writable-tmpfs -B $HOME/.ssh -B $PWD -B /mnt/ceph/users --pwd $PWD $IMAGE \
    python -m finfc.evaluate hydra.run.dir="${OUTPUT_PATH}/evaluate" \
        checkpoint_path="${OUTPUT_PATH}/train/lightning_logs/version_0/checkpoints/model.ckpt"
