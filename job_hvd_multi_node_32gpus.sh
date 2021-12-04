#!/bin/bash
#SBATCH --job-name horovod-multinode
#SBATCH -D /gpfs/home/nct01/nct01035/Lab10_horovod/
#SBATCH --output /gpfs/home/nct01/nct01035/Lab10_horovod/jobs/hvd_multinode_32gpus_%j.output
#SBATCH --error /gpfs/home/nct01/nct01035/Lab10_horovod/jobs/hvd_multinode_32gpus_%j.err

#SBATCH --ntasks-per-node=4     # must be added to create SLURM_NTASKS_PER_NODE var
#SBATCH -n 32                  # ntasks must be mutiple of ntasks-per-node. nnodes is ntasks/ntasks-per-node
#SBATCH -c 40                  # must be 160/ntasks-per-node. 160 is ncpus per node
#SBATCH --gres='gpu:4'          # must be ntasks-per-node
#SBATCH --time 00:55:00
##SBATCH --exclusive


module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML

HOSTS_FLAG="-H "
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  HOSTS_FLAG="$HOSTS_FLAG$node-ib0:$SLURM_NTASKS_PER_NODE,"
done
HOSTS_FLAG=${HOSTS_FLAG%?}


horovodrun --start-timeout 120 --gloo-timeout-seconds 120 -np $SLURM_NTASKS $HOSTS_FLAG --network-interface ib0 --gloo \
python3.7 tf2_keras_cifar_hvd.py --epochs 32 --batch_size 512 --model_name='resnet'

