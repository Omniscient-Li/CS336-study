"""Run the remaining GRPO experiment and record its command and GPU usage."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/a100_20261013"


def main():
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if usage:
        raise RuntimeError(f"GPU already has compute processes: {usage}")
    if (RUN / "hw7/grpo_history.json").exists():
        raise RuntimeError("hw7 already has results; use a new output directory for another experiment")
    command = [
        sys.executable, "-u", str(ROOT / "hw7/run_grpo_train_loop.py"),
        "--model_path", str(RUN / "hw4/checkpoint"),
        "--output_dir", str(RUN / "hw7"),
        "--n_grpo_steps", "20", "--learning_rate", "1e-5",
        "--rollout_batch_size", "64", "--group_size", "8",
        "--train_batch_size", "64", "--gradient_accumulation_steps", "64",
        "--epochs_per_rollout_batch", "2", "--loss_type", "grpo_clip",
        "--reward_mode", "strict", "--val_size", "128", "--val_every_steps", "5",
        "--sampling_max_tokens", "1024", "--seed", "42",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1", VLLM_USE_V1="0")
    for name in ("HF_HOME", "TORCH_HOME", "VLLM_CACHE_ROOT", "XDG_CACHE_HOME"):
        env[name] = str(ROOT / ".cache" / name.lower())
    metadata = {"command": command, "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (RUN / "hw7_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    stopped = threading.Event()

    def monitor():
        with (RUN / "hw7_gpu_usage.csv").open("w", encoding="utf-8") as output:
            output.write("elapsed_seconds,utilization_percent,memory_used_mib\n")
            while not stopped.is_set():
                reading = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits", "-i", "0",
                ], text=True).strip()
                output.write(f"{time.monotonic()-started:.2f},{reading}\n")
                output.flush()
                stopped.wait(2)

    started = time.monotonic()
    monitoring = threading.Thread(target=monitor)
    monitoring.start()
    try:
        with (RUN / "hw7.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        metadata.update(exit_code=completed.returncode, elapsed_seconds=time.monotonic()-started)
        (RUN / "hw7_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata), flush=True)
    finally:
        stopped.set()
        monitoring.join()
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
