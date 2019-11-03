#!/bin/bash

# WRN-16-2 => WRN-16-1
python3 train_zeroshot.py -tw 2 -td 16 -sw 1 -sd 16 -twgt ~/cifar10_WRN-16-2-seed10_model.204.h5 --seed 10
python3 train_zeroshot.py -tw 2 -td 16 -sw 1 -sd 16 -twgt ~/cifar10_WRN-16-2-seed23_model.204.h5 --seed 23
python3 train_zeroshot.py -tw 2 -td 16 -sw 1 -sd 16 -twgt ~/cifar10_WRN-16-2-seed45_model.204.h5 --seed 45
# ==========================================================================
# WRN-40-2 => WRN-40-1
python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 40 -twgt ~/cifar10_WRN-40-2-seed10_model.204.h5  --seed 10
python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 40 -twgt ~/cifar10_WRN-40-2-seed23_model.204.h5  --seed 23
python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 40 -twgt ~/cifar10_WRN-40-2-seed45_model.204.h5  --seed 45

# WRN-40-2 => WRN-16-1
python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 16 -twgt ~/cifar10_WRN-40-2-seed10_model.204.h5  --seed 10
python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 16 -twgt ~/cifar10_WRN-40-2-seed23_model.204.h5  --seed 23
python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 16 -twgt ~/cifar10_WRN-40-2-seed45_model.204.h5  --seed 45
# ==========================================================================

# WRN-40-2 => WRN-16-2
python3 train_zeroshot.py -tw 2 -td 40 -sw 2 -sd 16 -twgt ~/cifar10_WRN-40-2-seed10_model.204.h5  --seed 10
python3 train_zeroshot.py -tw 2 -td 40 -sw 2 -sd 16 -twgt ~/cifar10_WRN-40-2-seed23_model.204.h5  --seed 23
python3 train_zeroshot.py -tw 2 -td 40 -sw 2 -sd 16 -twgt ~/cifar10_WRN-40-2-seed45_model.204.h5  --seed 45
