
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
to run baseline methods, we provide scripts in methods folder.
to run a custom method, create python script following this template:
```
tokenizer = get_tokenizer("Llama-3.1-8B")
train_data_module = InstructJsonDataModule(
    train_data_path=train_data_path="/data/hf_cache/datasets/lsds_data/data/training_datatraining_data/tulu_3_v3.9_unfiltered.jsonl",
    batch_size=1, 
    tokenizer=get_tokenizer("Llama-3.1-8B"),
    max_seq_length=2048,
    subset_size=200000) 
train_data_module.setup()
train_dataset = train_data_module.train_dataset
train_dataloader = train_data_module.train_dataloader()

# Prepare validation dataloader
val_dataset = get_val_dataset("mmlu", tokenizer, seed=42, prompt_only=False, label_only=False)
val_dataloader = DataLoader(
    val_dataset, batch_size=8, pin_memory=True, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer),
)

# calculate score 
train_iterator = iter(train_dataloader)
for cnt, train_data in enumerate(tqdm(train_iterator), start=1):
    scores = user_defined_function()
    # save scores (as .npy or as hf_dataset)
```

## step 3 Selection, Training and Evaluation
```
bash train/run_finetune.sh configs/finetune_llama3-8b-tulu3.yaml
```