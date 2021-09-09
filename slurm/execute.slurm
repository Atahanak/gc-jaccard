#!/bin/bash
#SBATCH --job-name ipu-jac
#SBATCH -p ipuq    # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -c 32 # number of cores
#            #SBATCH --gres=ipu:1
#            #SBATCH --mem 128G # memory pool for all cores   # Bug in Slurm 20.02.5 torel@simula.no https://github.com/SchedMD/slurm/blob/master/NEWS
#SBATCH -o out/%j.out # STDOUT
#SBATCH -e out/%j.err # STDERR
#SBATCH -t 1-1:00 # time (D-HH:MM)

module purge
module load slurm/20.02.6
module load graphcore/vipu/1.12.6
#module load graphcore/vipu/1.8.1
module load graphcore/sdk/2.0.0
module load boost/1.73.0
#module load graphcore/gc/1.2.0

#export IPUOF_CONFIG_PATH=/cm/shared/apps/graphcore/vipu/etc/ipuof.conf.d/part_64_ipuof.conf             # 4 IPU Partition
#export IPUOF_CONFIG_PATH=/cm/shared/apps/graphcore/vipu/etc/ipuof.conf.d/p4_cl_a17_a18_ipuof.conf # 8 IPU partition

export IPUOF_CONFIG_PATH=/cm/shared/apps/graphcore/vipu/etc/ipuof.conf.d/p64_cl_a01_a16.conf
export POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./report"}'
#export POPLAR_ENGINE_OPTIONS='{"debug.instrumentCompute": "true", "debug.instrumentControlFlow": "true", "debug.computeInstrumentationLevel": "vertex"}'

cd ..
make
srun ./execs/jacc.out ./data/graphs/CiteSeersmol.bcsr 4
#srun time ./execs/jacc.out ./data/graphs/facebook_combined_c.bcsr 1
#srun ./execs/jacc.out ./data/graphs/com-dblp_c.bcsr
#GRAPHS=(
#   WikiCS
#   #CiteSeer
#   #Cora
#)
##   Cora
##   CiteSeer
#   #WikiCS
#   #CS
#   #PubMed
#   #Yelp
#   #Physics
#VERS=(
#  smol
#)

#EPOCHS=(
##10000
#1000
#)
#
#BATCHS=(
#10000
#)
##### LOGGING ####
##export POPLAR_LOG_LEVEL=TRACE
## export POPLAR_LOG_DEST= // can be stdout, stderr, or e file
##export POPLIBS_LOG_LEVEL=TRACE
## export POPLIBS_LOG_DEST=  // can be stdout, stderr, or e file
##srun ./execs/spmm.exe
##srun ./execs/ddmm.exe
#for GRAPH in ${GRAPHS[*]}
#do
#  for EPOCH in ${EPOCHS[*]}
#  do
#    for BATCH in ${BATCHS[*]}
#    do
#      srun  ~/GCGNN/execs/ppp.exe ~/graphs/uedgelist/${GRAPH}.graph ~/graphs/features/${GRAPH}.features ~/graphs/labels/${GRAPH}.labels ${EPOCH} ${BATCH}  ~/graphs/test_mask/${GRAPH}.test 
#      for  VER in ${VERS[*]}
#      do
#       echo "nvm"
#      #srun ~/GCGNN/execs/ppp.exe ~/graphs/uedgelist/${GRAPH}${VER}.graph ~/graphs/features/${GRAPH}${VER}.features ~/graphs/labels/${GRAPH}${VER}.labels ${EPOCH} ${BATCH}  ~/graphs/test_mask/${GRAPH}${VER}.test
#      done
#    done
#  done
#done
