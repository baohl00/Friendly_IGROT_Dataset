train_data=lasco
eval_data=circo # fiq, cirr, circo, dtin, sketchy, tuberlin, quickdraw
model=clip_large # clip_base, clip_large, blip(base)
encoder=both
batch_size=32
epochs=2
target_type=union
note=with_llava_${target_type}
comment=_5k_${note}_${encoder}
llava=without
#name=${model}_${epochs}_epo$comment
#name=final_model_blip_2_epo_5k_without_llava_union_both.pth
declare -a MODEL=("clip_base" "clip_large" "blip")
declare -a LLAVA=("with" "without") 
declare -a DATA=("cirr" "circo")

for model_i in "${MODEL[@]}"
do
	for llava_i in "${LLAVA[@]}"; 
	do
		for data_i in "${DATA[@]}"
		do
			name=final_model_${model_i}_2_epo_5k_${llava_i}_llava_${target_type}_both.pth
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
				--val_dataset $data_i \
				--val_load_path ${eval_load_path} \
				--submission_name ${name}
		done
	done
done
