# nano-vllm-spec-decode

在 nano-vllm 上加了两块：

- 投机解码：n-gram 和小模型草拟两种，支持贪心和随机采样，验证步走 CUDA Graph，按接受率动态选草拟长度 k。
- vLLM V1 式的混合批处理：prefill 和 decode 放在同一步里算，配合分段 CUDA Graph。

在 A100 + Qwen3-8B 上和 vLLM 0.15.1 用同一套客户端做了对比。设计、负面结果和每个数字的复现命令都在 [docs/REPORT.md](docs/REPORT.md)。

## 结果

A100-SXM4-80GB，目标模型 Qwen3-8B，草拟模型 Qwen3-0.6B，BF16。

| 场景 | 负载 | 结果 | 次数 |
|---|---|---|---|
| n-gram 动态 k vs 不投机 | 照抄型任务，温度 0，并发 1 | 88 → 335 tok/s，3.8× | 3 次均值 |
| 0.6B 草拟动态 k vs 不投机 | 照抄型任务，温度 0，并发 1 | 88 → 210 tok/s，2.4× | 3 次均值 |
| 0.6B 草拟动态 k vs 不投机 | 普通问答，温度 0，并发 1 | 88 → 116 tok/s，1.3× | 3 次均值 |
| 投机解码变慢的情况 | 普通问答，温度 0，并发 64 | n-gram 动态 k 0.94×，0.6B 草拟 k=4 0.79× | 1 |
| 混合批处理 vs vLLM，预算 2048 | ShareGPT 400 条，泊松 36.5 个/秒 | 吞吐 3565 vs 3688 tok/s（96.7%），TTFT P99 132 vs 190 ms | 3 次均值 |
| 混合批处理 vs vLLM，预算 2048 | ShareGPT 1000 条同时到达 | 吞吐 4197 vs 4333 tok/s（96.9%），decode 慢约 3% | 3 次均值 |
| 正确性 | 16 个变体，10 / 30 个 prompt | 全部通过 FP32 参考检查，eager 和图的四组对照 10/10、30/30 逐 token 一致 | 1 |

全部实验用 `bash scripts/run_p5.sh` 跑，同预算对比用 `bash scripts/run_p5.sh --sections env correctness budget --reuse <P5b 结果目录>`。单个数字对应的 `bench.py` 命令见 REPORT 第 4 节。

## 用法

```python
LLM(path, speculative_method="ngram", num_speculative_tokens=4)          # n-gram 投机解码
LLM(path, speculative_method="draft", draft_model=small_model_path)      # 小模型草拟
LLM(path, speculative_method="ngram", num_speculative_tokens=8, spec_dynamic_k=True)   # 动态 k，此时 k 是上限
LLM(path, enable_mixed_batching=True, max_num_batched_tokens=2048)       # 混合批处理
```

- `SamplingParams(temperature=0)` 是贪心采样，上游不允许。
- 贪心和随机采样都支持投机解码，输出和不开投机时等价。`llm.get_spec_stats()` 返回接受率等统计。只支持单卡。
- 一步里 decode 的请求超过 `spec_max_batch_size`（默认 512 // (1+k)）时，这一步不猜。
- 动态 k 每步在 0、1、2、4、8 里选，依据是各请求最近的接受率和启动时测出的每步耗时，接受率低时会自动不猜。
- 混合批处理默认走分段 CUDA Graph（`mixed_cuda_graph`）。`max_num_batched_tokens` 是每步 token 预算，A100 + 8B 建议至少 2048。暂时不能和小模型草拟一起开。
- 序列长度到 `max_model_len` 就结束。上游没有这个限制，在 CUDA Graph 下会越界。

常用命令（`MODEL` 指向模型目录，默认 `~/huggingface/Qwen3-0.6B/`）：

```bash
python tests/test_spec_unit.py                          # 投机解码单元测试（只用 CPU，不需要模型）
MODEL=/path/to/Qwen3-0.6B python tests/test_greedy.py   # 正确性测试
MODEL=/path/to/Qwen3-0.6B python bench.py               # 性能测试：TTFT、TPOT、吞吐
MODEL=/path/to/Qwen3-0.6B python bench.py --workload copy --concurrency 4 --spec ngram   # 投机解码
MODEL=/path/to/Qwen3-0.6B python scripts/bench_matrix.py                  # 整张对比表（约 1.5 小时）
MODEL=/path/to/Qwen3-0.6B python scripts/step_cost.py                     # 验证一步比普通一步贵多少
MODEL=/path/to/Qwen3-0.6B python scripts/profile_decode.py --spec ngram   # 一步 decode 里各部分的耗时
```

`bench.py` 新增了照抄型任务（`--workload copy`）、普通问答（`--workload chat`）、随机猜的最坏情况（`--spec-proposer random`）、并发上限（`--concurrency`），以及 `--spec draft --draft-model 路径`、`--dynamic-k`。

## 安装

需要 Python 3.10–3.12、CUDA GPU，依赖见 `pyproject.toml`（torch、triton、transformers、flash-attn、xxhash）。

```bash
pip install -e .
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
```

---

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（MIT 许可，提交 bb823b3），模型、分页 KV 缓存、前缀缓存等基础部分来自上游。
