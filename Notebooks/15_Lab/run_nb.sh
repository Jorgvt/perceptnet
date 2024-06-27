#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate testing_gpu

python -u 15_00_OnlyGabor.py
