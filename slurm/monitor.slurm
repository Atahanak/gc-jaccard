#!/bin/bash
#SBATCH --job-name monitor
#SBATCH -p ipuq    # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#            #SBATCH --gres=ipu:4
#            #SBATCH --mem 128G # memory pool for all cores   # Bug in Slurm 20.02.5 torel@simula.no https://github.com/SchedMD/slurm/blob/master/NEWS
#SBATCH -o out/%j.out # STDOUT
#SBATCH -e out/%j.err # STDERR
#SBATCH -t 1-1:00 # time (D-HH:MM)
#SBATCH


ulimit -s 10240
mkdir -p ~/output/ipuq

module purge
module load slurm/20.02.6
#module load graphcore/vipu/1.12.6
module load graphcore/vipu/1.8.1
module load graphcore/sdk/2.0.0
module load graphcore/gc/1.2.0

hostname
export IPUOF_CONFIG_PATH=/cm/shared/apps/graphcore/vipu/etc/ipuof.conf.d/p64_cl_a01_a16.conf
#export IPUOF_CONFIG_PATH=/cm/shared/apps/graphcore/vipu/etc/ipuof.conf.d/part_64_ipuof.conf             # 4 IPU Partition
#export IPUOF_CONFIG_PATH=/cm/shared/apps/graphcore/vipu/etc/ipuof.conf.d/p4_cl_a17_a18_ipuof.conf   # For GC-200 M2000 IPUPOD16
echo "Checking out occupation..."
#srun gc-info --list-devices
srun gc-info --list-devices
#srun gc-monitor
