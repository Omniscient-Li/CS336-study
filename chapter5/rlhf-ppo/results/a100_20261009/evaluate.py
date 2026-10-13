"""Compare initial and PPO policies on unseen IMDB test prompts."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, set_seed

root = Path(__file__).resolve().parent
device = "cuda"
set_seed(42)
torch.set_num_threads(4)
reward_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
classifier = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb").to(device).eval()
positive_id = next(int(k) for k, v in classifier.config.id2label.items() if v.upper() == "POSITIVE")
initial_tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
dataset = load_dataset("imdb", split="test")
dataset = dataset.filter(lambda x: len(x["text"]) > 200).shuffle(seed=2026).select(range(256))
rng = random.Random(2026)
prompts = [
    initial_tokenizer.decode(initial_tokenizer.encode(row["text"], add_special_tokens=False)[:rng.randrange(2, 8)])
    for row in dataset
]
results = {}
for label, path in [("initial", "lvwerra/gpt2-imdb"), ("ppo", str(root / "model"))]:
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    scores, probabilities, texts, responses = [], [], [], []
    for offset in range(0, len(prompts), 16):
        batch_prompts = prompts[offset:offset + 16]
        inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=16, do_sample=True, temperature=1.0,
                top_k=0, top_p=1.0, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            batch_responses = tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )
            batch_texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
            reward_inputs = reward_tokenizer(
                batch_texts, padding=True, truncation=True, max_length=512,
                return_token_type_ids=False, return_tensors="pt",
            ).to(device)
            logits = classifier(**reward_inputs).logits
        scores.extend(logits[:, positive_id].float().cpu().tolist())
        probabilities.extend(logits.softmax(-1)[:, positive_id].float().cpu().tolist())
        texts.extend(batch_texts)
        responses.extend(batch_responses)
    results[label] = {
        "positive_logit_mean": float(np.mean(scores)),
        "positive_probability_mean": float(np.mean(probabilities)),
        "positive_fraction": float(np.mean(np.array(probabilities) >= 0.5)),
        "positive_logits": scores,
        "positive_probabilities": probabilities,
        "responses": responses,
    }
    print(label, {k: v for k, v in results[label].items() if not isinstance(v, list)}, flush=True)
    del model
    torch.cuda.empty_cache()
summary = {
    "dataset": "imdb test split; reviews >200 characters; 256 prompts; seed 2026",
    "generation": "sampled; seed 42; temperature 1; top_k 0; top_p 1; max_new_tokens 16",
    "reward": "lvwerra/distilbert-imdb POSITIVE logits and softmax",
    "models": {label: {k: v for k, v in data.items() if not isinstance(v, list)} for label, data in results.items()},
    "positive_logit_mean_change": results["ppo"]["positive_logit_mean"] - results["initial"]["positive_logit_mean"],
    "positive_probability_mean_change": results["ppo"]["positive_probability_mean"] - results["initial"]["positive_probability_mean"],
    "note": "Single seeded sample per prompt; same classifier as training, not an independent quality benchmark.",
}
(root / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
examples = [
    {"prompt": prompt, "initial": results["initial"]["responses"][i], "ppo": results["ppo"]["responses"][i],
     "initial_positive_probability": results["initial"]["positive_probabilities"][i],
     "ppo_positive_probability": results["ppo"]["positive_probabilities"][i]}
    for i, prompt in enumerate(prompts)
]
(root / "evaluation_samples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
