DATA='val'

export CUDA_VISIBLE_DEVICES=0,1

python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=25040 \
	text_caption_generator.py \
	--split $DATA
