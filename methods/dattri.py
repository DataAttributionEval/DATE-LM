from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
import argparse
from datamodules.data_utils import *
from datamodules.load_data import *
from model_utils import *
from dattri.algorithm.influence_function import *
from dattri.algorithm.base import *
from dattri.task import AttributionTask
import torch.nn.functional as F
from omegaconf import OmegaConf


def get_lora_layers(model):
    """
    Extracts all LoRA parameter names and counts the total number of LoRA parameters.

    Args:
        model (torch.nn.Module): The model containing LoRA layers.

    Returns:
        tuple: (list of LoRA parameter names, total number of LoRA parameters)
    """
    lora_layers = []
    lora_cnt = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_layers.append(name)
            lora_cnt += param.numel()
    return lora_layers, lora_cnt


def get_lora_modules(model):
    """
    Extracts the module names corresponding to LoRA layers.

    Returns:
        list: List of LoRA module names used for EKFAC or LESS.
    """
    lora_modules = []
    for k, v in model.named_modules():
        if "lora" in k.lower() and "default" in k.lower() and "dropout" not in k.lower():
            lora_modules.append(k)
    return lora_modules


def get_loader(data, tokenizer):
    """
    Wraps raw input data into a PyTorch DataLoader using MessageDataset.

    Args:
        data (list): List of message dicts.
        tokenizer (AutoTokenizer): Tokenizer for encoding messages.

    Returns:
        DataLoader: Prepared loader for inference or training.
    """
    dataset = MessageDataset(data, tokenizer=tokenizer, max_length=1024)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader


def save_score(score, path):
    """
    Saves attribution scores as a torch file.

    Args:
        score (torch.Tensor or list): Attribution scores.
        path (str): Destination path for saving.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(score, path)


def attribute(args):
    """
    Main function to compute attribution scores using a specified method.

    Workflow:
    - Load model and tokenizer
    - Load dataset and format it into chat format
    - Define a differentiable loss function
    - Initialize attribution task and attributor based on method
    - Compute and cache gradients
    - Compute attribution scores from reference set to training data
    - Save scores to disk

    Args:
        args (OmegaConf.DictConfig): Configuration object loaded from YAML with keys:
            - method (str): Attribution method (e.g., DataInf, EKFAC, LESS, etc.)
            - task (str): Dataset/task name (e.g., ftrace, XSTest-response-Het)
            - subset (str): Optional subset or split
            - checkpoint (str): LoRA checkpoint path
            - base_model_path (str): Base pretrained model path
            - device (str): Device (e.g., "cuda" or "cpu")
            - regularization, damping, fim_estimate_data_ratio, proj_dim (float): Method-specific hyperparameters
            - save_path (str): Output path to save scores
    """
    tokenizer, model = checkpoints_load_func(None, args.checkpoint, args.base_model_path)
    train, ref = get_dataset(args.task, args.subset)
    train_chat = prepare_chat_format(train)
    ref_chat = prepare_chat_format(ref)

    def f(params, data_target_pair):
        input_ids, labels = data_target_pair
        outputs = torch.func.functional_call(model, params, input_ids, kwargs={
            "labels": labels.cuda()
        })
        return outputs.loss

    lora_layers, lora_cnt = get_lora_layers(model)
    lora_modules = get_lora_modules(model)

    train_loader = get_loader(train_chat, tokenizer)
    ref_loader = get_loader(ref_chat, tokenizer)

    task = AttributionTask(
        loss_func=f,
        model=model,
        base_model_path=args.base_model_path,
        checkpoints=args.checkpoint,
        checkpoints_load_func=checkpoints_load_func,
    )

    if args.method == "DataInf":
        attributor = IFAttributorDataInf(
            task=task,
            device=args.device,
            layer_name=lora_layers,
            regularization=args.regularization,
            fim_estimate_data_ratio=args.fim_estimate_data_ratio
        )
    elif args.method == "EKFAC":
        attributor = IFAttributorEKFAC(
            task=task,
            device=args.device,
            module_name=lora_modules,
            damping=args.damping,
        )
    elif args.method == "Grad_Dot":
        attributor = BaseInnerProductAttributor(
            task=task,
            device=args.device,
        )
    elif args.method == "Grad_Sim":
        attributor = BaseCosineSimilarityAttributor(
            task=task,
            device=args.device,
        )
    elif args.method == "LESS":
        attributor = IFAttributorLESS(
            task=task,
            layer_name=lora_layers,
            proj_dim=args.proj_dim,
            grad_in_dim=lora_cnt,
            device=args.device,
        )
    else:
        raise NotImplementedError(f"Attribution method '{args.method}' not implemented.")

    attributor.cache(train_loader)
    scores = attributor.attribute(train_loader, ref_loader)

    save_path = Path(args.save_path) / f"{args.method}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task == "Toxicity/Bias":
        scores = torch.mean(scores,dim = 1).tolist()
    elif args.task == "Counterfact" or args.task == "ftrace":
        scores = scores.transpose(0,1).tolist()
    else:
        raise NotImplementedError(f"Task '{args.task}' not implemented.")
    with open(save_path,"w") as f:
        json.dump(scores,f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    attribute(config)