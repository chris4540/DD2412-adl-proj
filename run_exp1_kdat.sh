#!/bin/bash

# Teacher wrn-40-2; Student wrn-16-1
python train_kdat.py --tdepth=40 --twidth=2 --tdepth=16 --swidth=1 \
                     -m 200 \
                     --tpath=saved_models/cifar10_WRN-40-2_model.h5 \
                     --dataset="cifar10" \
                     --savedir="./cifar10_T40-2_S16-1_seed10"
