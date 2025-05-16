from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import lightning as L
from litgpt.lora import GPT, Config, mark_only_lora_as_trainable
from litgpt.utils import instantiate_torch_optimizer, extend_checkpoint_dir
import yaml
from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (AutoTokenizer, AutoModelForCausalLM)


def load_lora_model_and_optimizer(
    fabric: L.Fabric,
    checkpoint_dir: Path,
    pretrained_checkpoint_dir: Optional[Path] = None,
    optimizer_config: Optional[Dict] = None,
    load_optimizer: bool = False,
) -> Tuple[GPT, Optional[torch.optim.Optimizer]]:
    """
    Loads the LoRA model and optionally the optimizer.

    Args:
        checkpoint_dir (Path): Path to the checkpoint directory with LoRA weights.
        pretrained_checkpoint_dir (Optional[Path]): Path to the base model checkpoint directory.
        optimizer_config (Optional[Dict]): Configuration for the optimizer.
        precision (Optional[str]): Precision setting for the model.
        load_optimizer (bool): Whether to load the optimizer state.

    Returns:
        Tuple[GPT, Optional[torch.optim.Optimizer]]: The loaded model and optimizer (if requested).
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    if pretrained_checkpoint_dir is not None:
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)

    # Initialize Fabric
    # fabric = L.Fabric(devices=1, precision=precision, accelerator=device)
    lora_params, meta_pretrained_checkpoint_dir, lora_precision = load_lora_metadata(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml", **lora_params)

    # Load the model
    # with fabric.init_module():
    model = GPT(config)


    # Load checkpoints
    lora_path = checkpoint_dir / "lit_model.pth.lora"
    pretrained_checkpoint = torch.load(str(pretrained_checkpoint_dir / "lit_model.pth"), mmap=True, weights_only=False)
    lora_checkpoint = torch.load(str(lora_path), mmap=True, weights_only=False)
    # pretrained_checkpoint = fabric.load(str(pretrained_checkpoint_dir / "lit_model.pth"), strict=False)
    # lora_checkpoint = fabric.load(str(lora_path), strict=False) 
    lora_checkpoint_model = lora_checkpoint.get("model", lora_checkpoint)

    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint_model)
    model.load_state_dict(pretrained_checkpoint, assign=True)
    mark_only_lora_as_trainable(model)

    optimizer = None
    if load_optimizer:
        assert optimizer_config is not None, "Optimizer config must be provided if load_optimizer is True."
        optimizer = instantiate_torch_optimizer(optimizer_config, model.parameters())
        lora_checkpoint_optimizer = lora_checkpoint.get("optimizer", None)
        if lora_checkpoint_optimizer is not None:
            optimizer.load_state_dict(lora_checkpoint_optimizer)
        else:
            print("Optimizer state not found in the checkpoint.")

    return model, optimizer

def checkpoints_load_func(model,checkpoint_path, base_model_path):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"}) # TODO changed
    base_model.resize_token_embeddings(len(tokenizer))
    new_model = load_peft_model(base_model, checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
    new_model.cuda()
    return new_model, tokenizer

def load_peft_model(base_model, checkpoint_path):
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    peft_model.config.pad_token_id = base_model.config.pad_token_id
    for name, param in peft_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    return peft_model

def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    indices = [i for i, (n,p) in enumerate(model.named_parameters()) if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in indices])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in indices])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq

def prepare_optimizer_state_hf(model, optimizer_state, device):
    # names = [n for n, p in model.named_parameters() if p.requires_grad]
    indices = [i for i, (n,p) in enumerate(model.named_parameters()) if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in range(len(indices))])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1) for n in range(len(indices))])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq

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

    #lora_params = {k: v for k, v in hparams.items() if k.startswith("lora_")}
    lora_params=hparams['lora']
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    precision = hparams.get("precision")
    return lora_params, pretrained_checkpoint_dir, precision