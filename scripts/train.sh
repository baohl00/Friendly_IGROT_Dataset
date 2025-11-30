train_data=lasco
val_data=lasco
model=clip_large
batch_size=32
epochs=1
comment=_10k
name=${model}_${epochs}_epo$comment
save_path=./ckpt/model_$name

TORCH_SHOW_CPP_STACKTRACES=1 
python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 \
	main.py \
	--dataset $train_data \
	--model $model \
	--batch-size $batch_size \
	--learning-rate 1e-4 \
   	--num-epochs $epochs \
	--save-path ${save_path} \
	--comment $name \
	--val_dataset $val_data
