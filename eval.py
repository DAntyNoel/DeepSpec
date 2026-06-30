from __future__ import annotations
import argparse
import json
import torch
from transformers import AutoConfig
from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.utils import CustomJSONEncoder

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]


def parse_task_specs(task_specs: str | None):
    if task_specs is None:
        return list(TASKS)
    tasks = []
    for raw_item in task_specs.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            name, raw_limit = item.split(":", 1)
            name = name.strip()
            max_samples = int(raw_limit)
        else:
            name = item
            max_samples = None
        if not name:
            raise ValueError(f"Invalid empty task in --tasks={task_specs!r}")
        tasks.append((name, max_samples))
    if not tasks:
        raise ValueError("--tasks must contain at least one task.")
    return tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--dataset-root", type=str, default="./eval_datasets")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=(
            "Comma-separated tasks with optional sample limits, e.g. "
            "'alpaca:8,gsm8k:8'. Defaults to the public DeepSpec suite."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    args = parser.parse_args()
    args.tasks = parse_task_specs(args.tasks)
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.device_count() <= 0:
        raise SystemExit("eval.py requires at least one visible CUDA device.")
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=torch.cuda.device_count(),
    )
