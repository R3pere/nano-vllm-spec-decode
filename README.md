> **本分支说明**：这是在上游 nano-vllm（提交 bb823b3）基础上做的个人学习项目，目标是加入投机解码和 KV swap，并测出真实收益。计划和进度见 [docs/PLAN.md](docs/PLAN.md)，每阶段做了什么见 [docs/DEVLOG.md](docs/DEVLOG.md)。下面 “Star History” 之前的内容是上游原文。
>
> 常用命令（`MODEL` 指向模型目录，默认 `~/huggingface/Qwen3-0.6B/`）：
>
> ```bash
> python tests/test_spec_unit.py                          # 投机解码单元测试（只用 CPU，不需要模型）
> MODEL=/path/to/Qwen3-0.6B python tests/test_greedy.py   # 正确性测试
> MODEL=/path/to/Qwen3-0.6B python bench.py               # 性能测试：TTFT、TPOT、吞吐
> MODEL=/path/to/Qwen3-0.6B python bench.py --workload copy --concurrency 4 --enforce-eager --spec ngram   # 投机解码
> ```
>
> 本分支对上游的改动：
> - `SamplingParams(temperature=0)` 表示贪心采样（上游不允许）。
> - n-gram 投机解码：`LLM(path, speculative_method="ngram", num_speculative_tokens=4)`。每一步先从前文里“照抄”猜几个 token，再让模型一次性检查，猜对的直接收下。贪心和随机采样都支持，输出与不开投机时等价。`llm.get_spec_stats()` 返回接受率等统计。目前检查这一步不走 CUDA Graph（P2 再做），只支持单卡。
> - `bench.py` 增加了“照抄型”任务（`--workload copy`）和并发上限（`--concurrency`）。

<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)