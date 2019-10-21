#!/bin/bash

python models.py > old.impl.txt
python net/wide_resnet.py > new.impl.txt