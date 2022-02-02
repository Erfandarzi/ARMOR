#!/usr/bin/env bash

GPU=$1

DATASET=$2

NOISE=$3

CLIENT_NUM=$4

python3 ./fed_train.py \
 --cuda $GPU \
 --dataset $DATASET \
 --global_noise_scale $NOISE \
 --client_num_in_total $CLIENT_NUM
