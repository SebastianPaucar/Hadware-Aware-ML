#!/bin/bash
#SBATCH --job-name=t4_job
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -A notchpeak-shared-short
#SBATCH -p notchpeak-shared-short
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:t4:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

nvidia-smi

module load apptainer/1.4.1
apptainer shell  --nv   --bind /uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-nehalem/gcc-8.5.0/cuda-12.5.0-7pt27rceb2uulofwak7zo3xkzspxgkg2:/usr/local/cuda ~/alma9-haml.sif
conda activate ha-ml
python ../l1_trigger_ae/trainer.py --train_model --config_path train_model/slave.yml
