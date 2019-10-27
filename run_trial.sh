#!/bin/bash


CUR_HEAD=`git rev-parse --short HEAD`
SAVE_FOLDER=zeroshot_${CUR_HEAD}
# Teacher wrn-40-2; Student wrn-16-1
python3 train_zeroshot_v2.py -tw 2 -td 40 -sw 1 -sd 16 \
    -tpath cifar10_WRN-40-2-seed45_model.172.h5 \
    --savedir ${SAVE_FOLDER}
