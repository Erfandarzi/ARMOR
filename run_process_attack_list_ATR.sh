#!/usr/bin/env bash

DATASET=$1

NOISE=$2

MODEL_NAME=$3

CLIENT_NUM=$4

python3 ./process_attack_list_ATR.py \
 --dataset $DATASET \
 --global_noise_scale $NOISE \
 --model_name $MODEL_NAME \
 --client_num_in_total $CLIENT_NUM
