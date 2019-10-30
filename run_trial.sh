#!/bin/bash


CUR_HEAD=`git rev-parse --short HEAD`
# SAVE_FOLDER=zeroshot_${CUR_HEAD}
SAVE_FOLDER=kdat_${CUR_HEAD}
# Teacher wrn-40-2; Student wrn-16-1
# python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 16 \
#     -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 \
#     --savedir ${SAVE_FOLDER} --seed 10

python3 train_kdat_v2.py --tdepth=40 --twidth=2 --sdepth=16 --swidth=2 \
                      -m 200 --savedir=${SAVE_FOLDER} \
                      --seed 45 \
                      -twghs ~/cifar10_WRN-40-2-seed45_model.172.h5 \
                      --dataset="cifar10"
