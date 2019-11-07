#!/bin/bash


CUR_HEAD=`git rev-parse --short HEAD`
SAVE_FOLDER=zeroshot-m100_${CUR_HEAD}
# SAVE_FOLDER=kdat_${CUR_HEAD}

python3 train_zeroshot.py --tdepth=40 --twidth=2 --sdepth=16 --swidth=2 \
                      -m 100 --savedir=${SAVE_FOLDER} \
                      --seed 23 \
                      -twgt ~/cifar10_WRN-40-2-seed23_model.204.h5 \
                      --dataset="cifar10"
