import argparse
import subprocess

from moment.common import PATHS

COMMANDS = [
    [
        "python3",
        "scripts/zero_shot/anomaly_detection.py",
        "--config_path=configs/anomaly_detection/zero_shot.yaml",
    ]
]


def get_task_name(command: str) -> str:
    script_path = command[1]
    if "anomaly_detection" in script_path:
        return "anomaly_detection"
    else:
        return "unknown_task"


def evaluate(
    path: str,
    opt_steps: int,
    run_name: str,
    gpu_id: int = 0,
) -> None:
    print(
        f"Evaluating: {path}",
        f"Steps: {opt_steps}",
        f"GPU ID: {gpu_id}\n",
    )
    pretraining_run_name = path.split("/")[-1]  # checkpoint dir
    for command in COMMANDS:
        command += [
            f'--run_name={run_name+"-"+get_task_name(command)}',
            f"--pretraining_run_name={pretraining_run_name}",
            f"--opt_steps={opt_steps}",
            f"--gpu_id={gpu_id}",
        ]
        print(f"Running command: {command}")
        subprocess.run(command)

    print(f"Evaluation finished for: {path} with {opt_steps} steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, help="Path to config file")
    parser.add_argument("--opt_steps", type=int, help="Number of optimization step")
    parser.add_argument("--run_name", type=str, help="Name of the W&B run")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    print("running eval script")
    print(args)
    evaluate(
        path=args.checkpoint_path,
        opt_steps=args.opt_steps,
        run_name=args.run_name,
        gpu_id=args.gpu_id,
    )
