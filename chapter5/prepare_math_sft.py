"""Join public MATH solutions onto the local train split, without changing splits."""

import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent


def main():
    repo = "DigitalLearningGmbH/MATH-lighteval"
    info = requests.get(f"https://huggingface.co/api/datasets/{repo}", timeout=60)
    info.raise_for_status()
    revision = info.json()["sha"]
    tree = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/data", timeout=60)
    tree.raise_for_status()
    solutions = {}
    for entry in tree.json():
        if entry["path"].endswith(".parquet") and "test" in entry["path"]:
            response = requests.get(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{entry['path']}", timeout=120)
            response.raise_for_status()
            for row in pq.read_table(io.BytesIO(response.content)).to_pylist():
                solutions[row["problem"]] = row["solution"]
    train = [json.loads(line) for line in (ROOT / "MATH/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for example in train:
        example["solution"] = solutions[example["problem"]]
    output = ROOT / "MATH/train_sft.jsonl"
    output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in train), encoding="utf-8")
    (ROOT / "MATH/solution_source.json").write_text(json.dumps({"repo": repo, "revision": revision, "matched": len(train)}, indent=2), encoding="utf-8")
    print(f"SFT solutions matched={len(train)} output={output}", flush=True)


if __name__ == "__main__":
    main()
