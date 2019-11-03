#!/bin/bash

#
#
#
# To run the result of Figure 2

# no teacher
python3 train_scratch.py  -w 1 -d 16 -m 20 --savedir=wrn-16-1-no-teacher-m20
python3 train_scratch.py  -w 1 -d 16 -m 40 --savedir=wrn-16-1-no-teacher-m40
python3 train_scratch.py  -w 1 -d 16 -m 60 --savedir=wrn-16-1-no-teacher-m60
python3 train_scratch.py  -w 1 -d 16 -m 80 --savedir=wrn-16-1-no-teacher-m80
python3 train_scratch.py  -w 1 -d 16 -m 100 --savedir=wrn-16-1-no-teacher-m100
# -----------------------------------------------------------------------------
# KD+AT
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 20  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed10_model.204.h5
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 40  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed10_model.204.h5
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 60  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed10_model.204.h5
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 80  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed10_model.204.h5
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 100  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed10_model.204.h5
# -----------------------------------------------------------------------------
