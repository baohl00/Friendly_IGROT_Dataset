DATA='val'
TRANSFORM='targetpad'

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25035 \
	data_utils.py \
	--data $DATA \
	--preprocess $TRANSFORM \
	--dimension 256
