#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate testing_gpu
exec_nb 13_00_Parametric.ipynb --dest "13_00_Parametric_exec.ipynb"
