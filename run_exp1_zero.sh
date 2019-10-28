#!/bin/bash
SEED=23
# Teacher wrn-40-2; Student wrn-16-1
python3 train_zeroshot.py -tw 1 -td 16 -sw 1 -sd 16
python3 train_zeroshot.py -tw 1 -td 16 -sw 1 -sd 16
python3 train_zeroshot.py -tw 1 -td 16 -sw 1 -sd 16
