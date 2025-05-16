CONFIG_FILE=$1

out_dir=$(yq .out_dir "$CONFIG_FILE" | tr -d '"')
checkpoint=$(yq .checkpoint "$CONFIG_FILE" | tr -d '"')
checkpoint_dir="$out_dir/$checkpoint"

train_config=$(yq .train_config "$CONFIG_FILE" | tr -d '"')
pretrained_checkpoint_dir=$(yq .checkpoint_dir "$train_config" | tr -d '"')

echo $checkpoint_dir
echo $pretrained_checkpoint_dir

python train/merge_lora.py $checkpoint_dir --pretrained_checkpoint_dir $pretrained_checkpoint_dir