#export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH

train_data=igrot # igrot, lasco, sketchy
eval_data=igrot # fiq, cirr, circo, dtin, sketchy, tuberlin, quickdraw, patterncom, pku
model=blip # clip_base, clip_large, blip(base), blip_flickr
encoder=both
batch_size=8
epochs=2
main_type=fuse
target_type=fuse
llava=with
data_amount=0
loss_type=triplet_infonce # infonce, extended_infonce, avg_infonce, triplet_infonce
note=${main_type}_${target_type}_${data_amount} #_${loss_type}
comment=_${train_data}_${note}
name=${model}_${epochs}_epo$comment

#name=final_model_${model}_2_epo_5k_${llava}_llava_${target_type}_both
save_path=./ckpt/final_model_$name
eval_load_path=./ckpt/final_model_$name.pth
python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=25057 \
	main.py \
	--training False \
	--dataset $train_data \
	--encoder $encoder \
	--model $model \
	--llava $llava \
	--main_type ${main_type} \
	--target_type ${target_type} \
	--data_amount $data_amount \
	--loss_type $loss_type \
	--vision_projector True \
	--batch-size $batch_size \
	--learning-rate 1e-4 \
	--num-epochs $epochs \
	--save-path ${save_path} \
	--comment $name \
	--inference True \
	--val_dataset $eval_data \
	--val_load_path ${eval_load_path} \
	--submission_name ${name}
