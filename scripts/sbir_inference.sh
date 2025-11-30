train_data=sketchy
eval_data=quickdraw # fiq, cirr, circo, dtin, sketchy, tuberlin, quickdraw
model=clip_base # clip_base, clip_large, blip(base)
encoder=both
batch_size=32
epochs=2
target_type=original
note=${target_type}
comment=_5k_${note}_${encoder}_fuse
#name=${model}_${epochs}_epo$comment
name=sbir_${model}_${epochs}_epo$comment
save_path=./ckpt/$name
eval_load_path=./ckpt/$name.pth
python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=25035 \
	main.py \
	--training False \
	--dataset $train_data \
	--encoder $encoder \
	--model $model \
	--type ${target_type} \
	--batch-size $batch_size \
	--learning-rate 1e-4 \
    --num-epochs $epochs \
    --save-path ${save_path} \
	--comment $name \
	--inference True \
	--val_dataset $eval_data \
	--val_load_path ${eval_load_path} \
	--submission_name ${name}
