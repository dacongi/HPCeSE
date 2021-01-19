To make this cupy code work, a V100 GPU is needed, 4 cores and 16 GB memory is enough to perform this task.

You should first create a python conda environment with cupy version 6.7.0 or later: 

conda install -c conda-forge cupy

then go to the directory and run the code with:

python run.py

You can also switch the input file by modifying the code line 18.
