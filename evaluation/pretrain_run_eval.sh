
set -ex

MODEL_PATH=$1

# evaluation metrics for pretrain
python evaluation/evaluate.py \
  ${MODEL_PATH} \
  --num_fewshot 0 \
  --batch_size 4 \
  --tasks "sciq,arc_easy,arc_challenge,logiqa,boolq,hellaswag,piqa,winogrande,openbookqa,lambada,lambada_openai" \
  --out_dir results_pretrain
