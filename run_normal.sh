#!/bin/bash

# run normal knowledge distillation
python3 train_kdat.py -td 40 -tw 2 -sd 16 -sw 1 -m 5000  --dataset=cifar10 -twgt ~/cifar10_WRN-40-2-seed45_model.204.h5 --seed 45 --beta 0
