export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use-env --nproc_per_node=2 --master_port=1145 train.py \
# -opt options/train/OPPO/train_sr_tsa.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use-env --nproc_per_node=2 --master_port=1145 train.py \
-opt options/train/Apple/train_sr_tsa.yml --launcher pytorch