#!/bin/bash

# Teacher wrn-40-2; Student wrn-16-1
# python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 16 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5  --seed 45
# python3 train_zeroshot.py -tw 2 -td 40 -sw 2 -sd 16 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5  --seed 45
# python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 40 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5  --seed 45

python3 train_zeroshot.py -tw 1 -td 40 -sw 1 -sd 16 -twgt ~/cifar10_WRN-40-1-seed45_model.200.h5 --seed 45
python3 train_zeroshot.py -tw 1 -td 40 -sw 2 -sd 16 -twgt ~/cifar10_WRN-40-1-seed45_model.200.h5 --seed 45

