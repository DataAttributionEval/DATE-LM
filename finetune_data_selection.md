
# Finetune data slection

## Step 1: Prepare datasets and models
download data:
```
bash data_processing/download_finetune_train_data.sh
bash data_processing/download_finetune_eval_data.sh
```
download model:
`litgpt download meta-llama/Llama-3.1-8B your_dir`

## Step 2: Run Data Scoring (skip for random selection)
We provide scripts for running baseline scoring methods in the `methods` folder. For example, to run gradient similarity:

`sbatch  methods/gradsim/probe_gradient_similarity_instruct_hf.sh`

To define a custom scoring function, you can copy `probe_gradient_similarity_instruct_hf.py` as template, and replace
`fit(fabric, state, train_data, val_data)` function with your own implementation


## Step 3: Selection, Training and Evaluation
```
bash train/run_finetune.sh configs/finetune_llama3-8b-tulu3.yaml
```