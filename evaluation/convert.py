from pathlib import Path
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
import os
import sys
import torch
from litgpt.utils import copy_config_files

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert.py <checkpoint_dir>")
        sys.exit(1)

    # Get the checkpoint directory from the command-line argument
    checkpoint_dir = Path(sys.argv[1])
    
    # Output directory for the converted Hugging Face model
    output_dir = checkpoint_dir  # Save in the same directory

    # Check if the conversion has already been done
    model_path = output_dir / "pytorch_model.bin"
    if model_path.exists():
        print(f"Conversion already completed. Hugging Face model files are in {output_dir}")
        return

    print(f"Converting LitGPT checkpoint from {checkpoint_dir} to Hugging Face format...")
    
    # Perform the conversion
    # copy_config_files(source_dir=checkpoint_dir, out_dir=output_dir)
    convert_lit_checkpoint(checkpoint_dir, output_dir)
    # Hack: LitGPT's conversion doesn't save a pickle file that is compatible to be loaded with
    # `torch.load(..., weights_only=True)`, which is a requirement in HFLM.
    # So we're `torch.load`-ing and `torch.save`-ing it again to work around this.
    state_dict = torch.load(output_dir / "model.pth", weights_only=False)
    torch.save(state_dict, model_path)
    # os.remove(output_dir / "model.pth") # delete litgpt model
    
    print(f"Conversion complete. Hugging Face model files saved in {output_dir}")

if __name__ == "__main__":
    main()