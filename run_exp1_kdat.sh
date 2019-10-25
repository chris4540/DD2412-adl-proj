#!/bin/bash

# Teacher wrn-40-2; Student wrn-16-1
python3 train_kdat.py --tdepth=40 --twidth=2 --sdepth=16 --swidth=1 \
                      -m 200 \
                      --teacher_weights="wrn-40-2-seed45/cifar10_WRN-40-2-seed45_model.172.h5" \
                      --dataset="cifar10"
