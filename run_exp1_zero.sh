#!/bin/bash
SEED=23
# Teacher wrn-40-2; Student wrn-16-1
python3 train_zeroshot.py -tw 1 -td 16 -sw 1 -sd 16 -tpath ~/wrn-16-1-seed23/cifar10_WRN-16-1-seed23_model.172.h5
python3 train_zeroshot.py -tw 1 -td 16 -sw 1 -sd 16 -tpath ~/wrn-16-1-seed23/cifar10_WRN-16-1-seed23_model.172.h5
python3 train_zeroshot.py -tw 1 -td 16 -sw 1 -sd 16 -tpath ~/wrn-16-1-seed23/cifar10_WRN-16-1-seed23_model.172.h5
