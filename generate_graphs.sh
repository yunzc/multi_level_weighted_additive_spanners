###===================
#!/bin/bash
#PBS -l select=1:ncpus=2:mem=12gb:pcmem=6gb -l walltime=5:00:00
#PBS -l cput=10:00:00
#PBS -q high_pri
#PBS -W group_list=mstrout
###-------------------

echo "Node name:"
hostname

cd /xdisk/kobourov/mig2020/extra/abureyanahmed/Graph_spanners
module load python/3.5/3.5.5
module load matlab/r2018b
python3 generate_multiple_graphs_with_same_setting.py

