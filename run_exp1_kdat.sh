#!/bin/bash

# Teacher wrn-40-2; Student wrn-16-1
python train_kdat.py --tdepth=40 --twidth=2 --sdepth=16 --swidth=1 \
                     -m 200 \
                     --teacher_weights=saved_models/cifar10_WRN-40-2_model.h5 \
                     --dataset="cifar10"
