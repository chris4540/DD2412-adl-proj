#!/bin/bash

# few shot
python3 belief_match.py -o fewshot.csv -s kdat-m200-cifar10_T40-2_S16-1_seed23_model.204.h5 -t cifar10_WRN-40-2-seed45_model.204.h5
# kd-at
python3 belief_match.py -o normal_kdat.csv -s kdat-m5000-cifar10_T40-2_S16-1_seed45_model.204.h5 -t cifar10_WRN-40-2-seed45_model.204.h5

# zeroshot
python3 belief_match.py -o zeroshot.csv -s cifar10-T40-2-S16-1-seed_45.model.79500.h5 -t cifar10_WRN-40-2-seed45_model.204.h5
