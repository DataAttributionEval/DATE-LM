
# Training Data Selection

Example config: [configs/demo.yaml](configs/demo.yaml)
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

### Step 1: Data Processing
tokenize and process training data and reference data

```sbatch data_processing/run_preprocess.sh $config_path```

### Step 2: Data Scoring
produces score for each datapoint. higher = better. This step can be skipped for random selection.

We provide scripts for each baseline method, that users can use as templates, which can be found in methods folder. 


### Step 3: Data Selection based on scores and diversity metric
normalize scores
run diversity metric (gumbel)
output selected data
```python methods/select_data.py --config "$config_path"```

### Step 4: Training & Evaluation
```sbatch train/run_pretrain.sh $config_path```


