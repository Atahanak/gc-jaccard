#!/bin/bash
#SBATCH --job-name ipu-jac
#SBATCH -p ipuq    # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#            #SBATCH --gres=ipu:4
#            #SBATCH --mem 128G # memory pool for all cores   # Bug in Slurm 20.02.5 torel@simula.no https://github.com/SchedMD/slurm/blob/master/NEWS
#SBATCH -o out/%j.out # STDOUT
#SBATCH -e out/%j.err # STDERR
#SBATCH -t 1:00 # time (D-HH:MM)

module purge
module load slurm/20.02.6

hostname
cat /proc/cpuinfo
