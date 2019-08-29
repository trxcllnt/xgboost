# please run with '. this-file-name.sh' or 'source this-file-name.sh'

conda activate cuda10.0
export JAVA_HOME=/opt/conda
export NCCL_ROOT=/usr/local/nccl
export RMM_ROOT=/opt/conda/envs/cuda10.0
rm -f /usr/local/cuda /usr/local/nccl
ln -s cuda-10.0 /usr/local/cuda
ln -s nccl_2.4.7-1+cuda10.0_x86_64 /usr/local/nccl
