"""ShareGPT workload: the dataset vLLM's own benchmarks use, sampled the same way.

  download  fetch ShareGPT_V3_unfiltered_cleaned_split.json (HuggingFace dataset
            anon8231489123/ShareGPT_Vicuna_unfiltered, ~673MB). Set HF_ENDPOINT=https://hf-mirror.com
            where huggingface.co is slow or blocked.
  sample    turn it into a list of requests and save it as JSON:
              [{"prompt_token_ids": [...], "output_len": N}, ...]
            Like vLLM's benchmark (ShareGPTDataset in vllm/benchmarks/datasets.py): shuffle the conversations
            with a fixed seed, keep those with at least two turns, the first turn is the prompt and the
            number of tokens of the second turn (the answer) is the output length. Skip a request when the
            prompt or the answer has fewer than 4 tokens, the prompt more than 1024, or both together more
            than 2048. The prompt is the raw text, no chat template (vLLM does the same).
            bench.py runs it with ignore_eos, so every engine generates exactly output_len tokens and both
            engines get exactly the same input tokens.

Usage: python scripts/sharegpt.py download --out DIR
       python scripts/sharegpt.py sample --dataset ShareGPT_V3_unfiltered_cleaned_split.json --model PATH
                                         --num N [--seed 0] --out sample.json
"""
import argparse
import hashlib
import json
import os
import random

REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
SHA256 = "35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"
MIN_LEN, MAX_PROMPT_LEN, MAX_TOTAL_LEN = 4, 1024, 2048


def download(out):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset", local_dir=out)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != SHA256:
        raise SystemExit(f"{path}: sha256 {h.hexdigest()}, expected {SHA256} (incomplete download?)")
    print(path)


def sample(dataset, tokenizer, num, seed=0):
    with open(dataset) as f:
        convs = [c["conversations"] for c in json.load(f) if len(c.get("conversations", [])) >= 2]
    random.Random(seed).shuffle(convs)
    requests = []
    for conv in convs:
        prompt = tokenizer(conv[0]["value"]).input_ids
        output_len = len(tokenizer(conv[1]["value"]).input_ids)
        if len(prompt) < MIN_LEN or output_len < MIN_LEN or len(prompt) > MAX_PROMPT_LEN or len(prompt) + output_len > MAX_TOTAL_LEN:
            continue
        requests.append({"prompt_token_ids": prompt, "output_len": output_len})
        if len(requests) == num:
            return requests
    raise SystemExit(f"only {len(requests)} usable conversations, {num} wanted")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--out", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--dataset", required=True)
    s.add_argument("--model", required=True, help="its tokenizer turns the text into token ids")
    s.add_argument("--num", type=int, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cmd == "download":
        download(args.out)
        return
    from transformers import AutoTokenizer
    requests = sample(args.dataset, AutoTokenizer.from_pretrained(args.model), args.num, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(requests, f)
    prompt_lens = [len(r["prompt_token_ids"]) for r in requests]
    output_lens = [r["output_len"] for r in requests]
    print(f"{len(requests)} requests, prompt tokens mean {sum(prompt_lens) / len(requests):.0f} (max {max(prompt_lens)}), "
          f"output tokens mean {sum(output_lens) / len(requests):.0f} (max {max(output_lens)}) -> {args.out}")


if __name__ == "__main__":
    main()
