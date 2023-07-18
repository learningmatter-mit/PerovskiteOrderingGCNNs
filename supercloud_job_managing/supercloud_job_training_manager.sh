#!/bin/bash


module load anaconda/2021a
source activate Perovskite_ML_Environment

python ML_job_managers.py $LLSUB_RANK $LLSUB_SIZE