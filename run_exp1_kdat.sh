#!/bin/bash

# # Teacher wrn-40-2; Student wrn-16-1
# python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 10
# python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 23
# # python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 45


#Teacher wrn-16-2; Student wrn-16-1
python3 train_kdat.py -td 16 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-16-2-seed45_model.181.h5 --seed 10
python3 train_kdat.py -td 16 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-16-2-seed45_model.181.h5 --seed 23
python3 train_kdat.py -td 16 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-16-2-seed45_model.181.h5 --seed 45

# Teacher wrn-40-2; Student wrn-40-1
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 10
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 23
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 45

# Teacher wrn-40-2; Student wrn-40-1
python3 train_kdat.py -td 40 -tw 2 -sd 40 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 10
python3 train_kdat.py -td 40 -tw 2 -sd 40 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 23
python3 train_kdat.py -td 40 -tw 2 -sd 40 -sw 1 -m 200  --dataset=cifar10 -twgts ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 45
