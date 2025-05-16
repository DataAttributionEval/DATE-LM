# gpu_index=0
# for s in {0..7}; do
#     echo $s
#     CUDA_VISIBLE_DEVICES=$gpu_index python mates/predict_data_influence.py --ckpt $ckpt --shard $s 8 > log_job_s${s}_gpu${gpu_index}.out 2>&1 &
#     ((gpu_index=(gpu_index+1)%8))
# done

# model_dir="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/10000-data_influence_model-flan"
# output_dir="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/mates_flan"
# select_from_size=1024000

# python methods/mates/predict_data_influence.py \
#     --model_dir $model_dir \
#     --output_dir $output_dir \
#     --select_from_size $select_from_size \
#     --shard 0 8

# SMALL TEST
model_dir="/data/group_data/cx_group/out/pythia-1b/fineweb/sample-350BT/30000-data_influence_model-lambada"
output_dir="/data/user_data/emilyx/mates_flan_30"
select_from_size=1024
python methods/mates/predict_data_influence.py \
    --model_dir $model_dir \
    --output_dir $output_dir \
    --select_from_size $select_from_size \
    --shard 0 8
