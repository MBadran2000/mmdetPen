#!/bin/bash

#SBATCH --job-name=mmdetection # job name

#SBATCH --output=jobs/%j%x.out 
                                             # output log file
#SBATCH --error=jobs/%j%x.err 
                                              # error file
#SBATCH --time=24:00:00  

#SBATCH --nodes=1    

#SBATCH --partition=gpu 

#SBATCH --ntasks=40       

#SBATCH --gres=gpu:1   

%module load cuda
source ~/.bashrc

conda activate pengwin
cd /home/y.nawar/mmdetection/

python /scratch/dr/m.badran/pengwin/Nawar/mmdetection/tools/train.py \
    /scratch/dr/m.badran/pengwin/Nawar/configs/config_mask_rcnn.py