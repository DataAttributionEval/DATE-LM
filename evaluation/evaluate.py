# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# This is a modified version of the original file, which can be found at:
# site-packages/litgpt/eval/evaluate.py


import json
import os
from pathlib import Path
from typing import Optional, Union
import torch

from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import CLI, copy_config_files


def prepare_results(results, tasks, save_filepath, print_results=True):
    for filename in os.listdir(save_filepath.parent):
        file_path = os.path.join(save_filepath.parent, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    from lm_eval.utils import make_table

    if print_results:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

    json_result = json.dumps(results["results"], indent=2, ensure_ascii=False)
    save_filepath.open("w", encoding="utf-8").write(json_result)

    with open(save_filepath.parent / "results.txt", "w") as fw:
        for task in tasks:
            if "acc_norm,none" in results["results"][task]:
                fw.write(
                        "{:.3f} ± {:.3f}".format(
                            results["results"][task]["acc_norm,none"],
                            results["results"][task]["acc_norm_stderr,none"],
                        )
                        + "\t"
                    )
            else:
                fw.write(
                    "{:.3f} ± {:.3f}\t".format(
                        results["results"][task]["acc,none"],
                        results["results"][task]["acc_stderr,none"],
                    )
                )

def convert_and_evaluate(
    checkpoint_dir: Path,
    tasks: Optional[str] = None,
    out_dir: Optional[Path] = None,
    force_conversion: bool = False,
    num_fewshot: Optional[int] = None,
    batch_size: int = 1,
    device: Optional[str] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    limit: Optional[float] = None,
    seed: int = 1234,
    save_filepath: Optional[Path] = None,
) -> None:
    """Convert a LitGPT model and run the LM Evaluation Harness

    Arguments:
        checkpoint_dir: Directory where the `lit_model.pth` and tokenizer files are located.
        out_dir: Directory in which to save the converted checkpoints for evaluation.
            Saves to `checkpoint_dir`/evaluate by default.
        force_conversion: Set to `True` to reconvert the model and override
            an existing model.pth from a previous evaluation call.
        tasks: CSV of task names to evaluate. Example: "hellaswag,truthfulqa_mc2,mmlu"
        num_fewshot: Number of examples in few-shot context.
        batch_size: Batch size configuration.
        device: Device to use for evaluation, for example, "cuda" or "cuda:0".
        limit: Limit on number of examples per task.
        seed: Random seed.
        save_filepath: The file where the results will be saved.
            Saves to `out_dir/results.json` by default.
    """

    from lm_eval import evaluator

    if tasks is None:
        from lm_eval.tasks import TaskManager

        taskm = TaskManager()
        print("\n".join(taskm.task_index.keys()))
        print(
            "\n\nTo evaluate multiple tasks, you can chain the task names "
            "listed above via a comma-separated list."
            "\nFor example: `--tasks 'hellaswag,truthfulqa_mc2,mmlu'`. "
            "\nTo search for a specific task, use `litgpt evaluate | grep task_name`."
        )
        return

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(checkpoint_dir)

    if out_dir is None:
        out_dir = checkpoint_dir / "evaluate"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_filepath = (
        out_dir / Path("results.json") if save_filepath is None else Path(save_filepath)
    )

    model_path = out_dir / "pytorch_model.bin"
    if not model_path.exists() or force_conversion:
        copy_config_files(source_dir=checkpoint_dir, out_dir=out_dir)
        convert_lit_checkpoint(checkpoint_dir=checkpoint_dir, output_dir=out_dir)

        # Hack: LitGPT's conversion doesn't save a pickle file that is compatible to be loaded with
        # `torch.load(..., weights_only=True)`, which is a requirement in HFLM.
        # So we're `torch.load`-ing and `torch.sav`-ing it again to work around this.
        state_dict = torch.load(out_dir / "model.pth", weights_only=False)
        torch.save(state_dict, model_path)
        os.remove(out_dir / "model.pth")

    from lm_eval.models.huggingface import HFLM

    model = HFLM(
        pretrained=str(out_dir.resolve()),
        device=device,
        batch_size=batch_size,
        dtype=dtype,
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks.split(","),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    prepare_results(results, tasks.split(","), save_filepath)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI
    CLI(convert_and_evaluate)