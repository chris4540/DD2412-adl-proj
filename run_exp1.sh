#!/bin/bash
# WRN-40-2
python3 train_scratch.py  -w 2 -d 40 --seed=10 --savedir=wrn-40-2-seed10
python3 train_scratch.py  -w 2 -d 40 --seed=23 --savedir=wrn-40-2-seed23
python3 train_scratch.py  -w 2 -d 40 --seed=45 --savedir=wrn-40-2-seed45

# WRN-40-1
python3 train_scratch.py  -w 1 -d 40 --seed=10 --savedir=wrn-40-1-seed10
python3 train_scratch.py  -w 1 -d 40 --seed=23 --savedir=wrn-40-1-seed23
python3 train_scratch.py  -w 1 -d 40 --seed=45 --savedir=wrn-40-1-seed45

# WRN-16-2
python3 train_scratch.py  -w 2 -d 16 --seed=10 --savedir=wrn-16-2-seed10
python3 train_scratch.py  -w 2 -d 16 --seed=23 --savedir=wrn-16-2-seed23
python3 train_scratch.py  -w 2 -d 16 --seed=45 --savedir=wrn-16-2-seed45

# WRN-16-1
python3 train_scratch.py  -w 1 -d 16 --seed=10 --savedir=wrn-16-1-seed10
python3 train_scratch.py  -w 1 -d 16 --seed=23 --savedir=wrn-16-1-seed23
python3 train_scratch.py  -w 1 -d 16 --seed=45 --savedir=wrn-16-1-seed45
