#!/bin/bash


python3 train_zeroshot.py -tw 2 -td 40 -sw 1 -sd 16 --seed 23 -m 100 \
                          -twgt ~/cifar10_WRN-40-2-seed23_model.204.h5
