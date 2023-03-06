#!/bin/bash

# source /home/amueller/miniconda3/bin/activate
# conda activate lm-eval

model_path=$1
model_type=$2
task_type=$3
num_fewshot=$4

CUDA_VISIBLE_DEVICES=`free-gpu` python babylm_eval.py $model_path $model_type -t $task_type -n $num_fewshot
