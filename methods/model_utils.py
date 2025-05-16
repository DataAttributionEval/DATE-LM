import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM)
from peft import PeftModel, LoraConfig, get_peft_model
import os
import torch
from litgpt.lora import GPT, Config, lora_filter, merge_lora_weights
from litgpt.utils import extend_checkpoint_dir
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Tuple
from transformers import AutoConfig, AutoModelForCausalLM
import lightning as L
import torch
import yaml

def load_lora_metadata(checkpoint_dir: Path) -> Tuple[Dict[str, Any], Path, Optional[str]]:
    hparams_file = checkpoint_dir / "hyperparameters.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError(
            f"The path {str(hparams_file)!r} is not a valid checkpoint directory. It is missing a"
            f" `hyperparameters.yaml` file. Please point to the checkpoint directory that was produced by"
            f" the `litgpt/finetune/lora.py` script."
        )

    with open(hparams_file, "r", encoding="utf-8") as file:
        hparams = yaml.safe_load(file)

    # Grab the nested `lora` dictionary if it exists
    lora_params = hparams.get("lora", {})

    # Extract other metadata
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    precision = hparams.get("precision")

    return lora_params, pretrained_checkpoint_dir, precision

def load_tokenizer_and_base_model(model_id="meta-llama/Llama-3.2-1B"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    base_model.resize_token_embeddings(len(tokenizer))
    return tokenizer, base_model

def load_peft_model(base_model, checkpoint_path):
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    peft_model.config.pad_token_id = base_model.config.pad_token_id
    for name, param in peft_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    return peft_model

def checkpoints_load_func(model,checkpoint_path,base_model_path):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    base_model.resize_token_embeddings(len(tokenizer))
    new_model = load_peft_model(base_model, checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
    new_model.cuda()
    return tokenizer, new_model

def checkpoints_load_func_litgpt(model,checkpoint,base_model_path):
    checkpoint = Path(checkpoint)
    base_model_path = Path(base_model_path)
    lora_params, meta_pretrained_checkpoint_dir, lora_precision = load_lora_metadata(checkpoint)
    print(lora_params)
    print(checkpoint)
    config = Config.from_file(checkpoint / "model_config.yaml", **lora_params)
    pretrained_checkpoint = torch.load(str(base_model_path / "lit_model.pth"), mmap=True)
    lora_path = checkpoint / "lit_model.pth.lora"
    lora_checkpoint = torch.load(str(lora_path), mmap=True)
    lora_checkpoint = lora_checkpoint.get("model", lora_checkpoint)
    model = GPT(
        config,
    )
    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint)
    model.load_state_dict(pretrained_checkpoint, assign=True)
    model.float()
    model.cuda()
    model.eval()
    lora_trainable_weights = 0
    for name, param in model.named_parameters():
        print(name)
        print(type(param))
        print(param.shape)
        print("="*100)
        if "lora" in name.lower():
            param.requires_grad = True
            lora_trainable_weights += param.numel()
        else:
            param.requires_grad = False
    print(f"Trainable lora parameters: {lora_trainable_weights}")
    return model

def checkpoints_load_func_litgpt_EKFAC(model,checkpoint,base_model_path):
    checkpoint = Path(checkpoint)
    base_model_path = Path(base_model_path)
    lora_params, meta_pretrained_checkpoint_dir, lora_precision = load_lora_metadata(checkpoint)
    config = Config.from_file(checkpoint / "model_config.yaml", **lora_params)
    pretrained_checkpoint = torch.load(str(base_model_path / "lit_model.pth"), mmap=True)
    lora_path = checkpoint / "lit_model.pth.lora"
    lora_checkpoint = torch.load(str(lora_path), mmap=True)
    lora_checkpoint = lora_checkpoint.get("model", lora_checkpoint)
    model = GPT(
        config,
    )
    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint)
    model.load_state_dict(pretrained_checkpoint, assign=True)
    model.float()
    model.cuda()
    lora_trainable_weights = 0
    for name, param in model.named_parameters():
        print(name)
        print("="*100)
        if "lora" in name.lower():
            param.requires_grad = True
            lora_trainable_weights += param.numel()
        else:
            param.requires_grad = False
    print(f"Trainable lora parameters: {lora_trainable_weights}")
    return model






