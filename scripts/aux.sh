#export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH

train_data=lasco
eval_data=cirr # fiq, cirr, circo, dtin, sketchy, tuberlin, quickdraw
model=blip # clip_base, clip_large, blip(base)
encoder=both
batch_size=32
epochs=2
target_type=original
note=with_llava_${target_type}
comment=_5k_${note}_${encoder}
#name=${model}_${epochs}_epo$comment
#name=final_model_blip_2_epo_5k_with_llava_union_both_2025-04-04-15-29-13.pth
name=final_model_blip_2_epo_5k_without_llava_original_both_2025-04-01-22-18-58.pth
save_path=./ckpt/final_model_$name
eval_load_path=./ckpt/$name
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
