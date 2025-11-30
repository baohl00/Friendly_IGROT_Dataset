DATA='val'
TRANSFORM='targetpad'

export CUDA_VISIBLE_DEVICES=0,1

python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=25030 \
	data_llava_utils.py \
	--data $DATA \
	--preprocess $TRANSFORM \
	--dimension 256
