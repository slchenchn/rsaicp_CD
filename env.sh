# change conda environment and cuda version
cd ~/code/rsaicp_CD
# source ~/.bashrc
source ~/shell_tools/switch_cuda.sh 11.0

# export CUDA_HOME=:/usr/local/cuda-11.0
# export LD_LIBRARY_PATH=:/usr/local/cuda-11.0/lib64
# export PATH=:/usr/local/cuda-11.0/bin

# conda init bash
conda activate mmlab-cu110