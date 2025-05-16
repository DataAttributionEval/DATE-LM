
# Training Data Selection

Example config: [configs/pretrain_random_minimal.yaml](configs/pretrain_random_minimal.yaml)
```yaml
# data preprocessing
train_dataset_name: fineweb 
reference_dataset_name: lambada

## data selection
method: random
gumbel_temp: 0.5
select_from_size: 1024000
selection_size: 102400

## training and evaluation
model_class: pythia
train_config: configs/pretrain_decay_pythia-1b.yaml
```

### Step 1: Prepare datasets and models
tokenize and process training data and reference data:

```
python data_processing/process_fineweb.py --base_dir [your_base_dir]
python data_processing/prepare_lambada.py --base_dir [your_base_dir]
```

download our custom-trained pythia-1b model on fineweb:

```
huggingface-cli download DataAttributionEval/Pythia-1b-fineweb-random-30k --revision main --cache-dir [your_base_dir]/out/pythia-1b/fineweb/sample-350BT/random/step-00030000/

huggingface-cli download DataAttributionEval/Pythia-1b-fineweb-random-10k --revision main --cache-dir [your_base_dir]/out/pythia-1b/fineweb/sample-350BT/random/step-00010000/
```


### Step 2: Run Data Scoring (skip for random selection)
We provide scripts for running baseline scoring methods in the `methods` folder. For example, to run gradient similarity:

`sbatch methods/gradsim/probe_gradsim_lambada_array.sh [your_base_dir]`

To define a custom scoring function, you can copy `probe_gradient_similarity.py` as template, and replace
`fit(fabric, state, train_data, val_data)` function with your own implementation


### Step 3: Run Data Selection with Gumbel-top-k
```bash methods/run_selection.sh configs/pretrain_gradsim.yaml```
change appropiate parameters if running with custom method

### Step 4: Training & Evaluation
```sbatch train/run_pretrain.sh configs/pretrain_gradsim.yaml```

you can find eval metric results in out_dir/results.txt



