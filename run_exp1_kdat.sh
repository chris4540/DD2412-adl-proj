#!/bin/bash

set -e

# 16-2 => 16-1 Row 1
# python3 train_kdat.py -td 16 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-16-2-seed45_model.181.h5 --seed 10
# python3 train_kdat.py -td 16 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-16-2-seed45_model.181.h5 --seed 23
# python3 train_kdat.py -td 16 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-16-2-seed45_model.181.h5 --seed 45

# 40-2 => 16-1; Row3
# python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 10
# python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 23
# python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 45

# 40-2 => 40-1; Row 6
# python3 train_kdat.py -td 40 -tw 2 -sd 40 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 10
# python3 train_kdat.py -td 40 -tw 2 -sd 40 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 23
# python3 train_kdat.py -td 40 -tw 2 -sd 40 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 45

# ============================================================================================================================
# 40-1 => 16-1; Row 2
python3 train_kdat.py -td 40 -tw 1 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-1-seed45_model.200.h5 --seed 10
python3 train_kdat.py -td 40 -tw 1 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-1-seed45_model.200.h5 --seed 23
python3 train_kdat.py -td 40 -tw 1 -sd 16 -sw 1 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-1-seed45_model.200.h5 --seed 45

# 40-1 => 16-2; Row 4
python3 train_kdat.py -td 40 -tw 1 -sd 16 -sw 2 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-1-seed45_model.200.h5 --seed 10
python3 train_kdat.py -td 40 -tw 1 -sd 16 -sw 2 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-1-seed45_model.200.h5 --seed 23
python3 train_kdat.py -td 40 -tw 1 -sd 16 -sw 2 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-1-seed45_model.200.h5 --seed 45

# 40-2 => 16-2 Row 5
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 2 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 10
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 2 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 23
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 2 -m 200  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.172.h5 --seed 45
