"""P5: every measurement for the rented GPU, in one resumable script (scripts/run_p5.sh calls it).

Every run is its own process (a fresh engine each time) and leaves <out>/<name>.json plus <name>.log.
A run whose JSON already exists is skipped, so after an interruption (or a lost instance with /data kept)
just start the script again. A run that crashed or timed out leaves no JSON and is tried again next time;
a correctness test that ran but failed does leave one ({"ok": false}) and is not rerun. Delete a JSON
to measure that run again.

Order (most important first, in case time runs out):
  env          versions, GPU, driver, CPU, pip freeze of both venvs -> <out>/env/
  correctness  tests/test_greedy.py on the target model, draft variants with the small draft model; if it fails,
               the script stops here (--keep-going to measure anyway)
  sharegpt     ShareGPT (sampled like vLLM's benchmark, sample saved as <out>/sharegpt_sample.json), nano-vllm
               against vLLM: all requests at once (throughput) and Poisson arrivals at two rates (near
               saturation, overloaded; both found from the all-at-once run of nano-vllm), no speculation and
               n-gram k=4 on both engines, plus nano-vllm's mixed batching (budget 512). Draft model k=4 once.
  spec         copy / chat workloads, temperature 0 and 0.6, concurrency 1/4/16/64: CUDA graph, n-gram
               dynamic k, draft model k=4, draft model dynamic k
  mixed        long prompts arriving as a Poisson process (P4's workload, more requests): first an
               all-at-once run gives the request rate the GPU can sustain, then 0.9x and 1.5x of it;
               upstream scheduling, mixed 512, mixed 2048, mixed 512 + n-gram dynamic k
  step_cost    scripts/step_cost.py break-even tables, n-gram and draft
  repeats      the key cells (all ShareGPT cells; spec copy/chat T0 c1 and copy T0 c16; mixed graph and
               mixed512 at both rates) run twice more, after everything else, so drift over the session
               shows up as spread instead of bias
  budget       (P5b3, not in the default list) ShareGPT, nano-vllm's mixed batching at token budgets 2048 / 4096 /
               8192 / 16384 against vLLM at the same budgets (P5b only had nano at 512, vLLM at 16384), all at once
               and at P5b's two Poisson rates; budgets 2048 and 16384 repeated --repeats times. Then once each:
               nano-vllm's upstream scheduling (to tie this session to P5b's table) and mixed 2048 + n-gram
               dynamic k; then scripts/profile_mixed.py at budgets 512 and 2048 (phases of a mixed step, in the
               log); then the repeats. Before it, correctness30: tests/test_greedy.py --num-prompts 30 on the eager/graph pairs
               (decode batches up to 30, the case that hid the P5b mismatch), and scripts/check_padding.py.
Then everything is packed into results/p5_<date>.tar.gz. --table-only prints the summary tables.
--reuse DIR copies rates.json and sharegpt_sample.json from an earlier result folder (P5b's), so the Poisson rates
and the requests are exactly the same as there.

--smoke shrinks every run (few requests, short outputs) so the whole script can be rehearsed on a small GPU.

Usage: python scripts/run_p5.py --model PATH --draft-model PATH --vllm-python PATH --sharegpt RAW.json
                                [--out results/p5] [--smoke] [--sections ...] [--repeats 3] [--dry-run] [--table-only]
P5b3:  bash scripts/run_p5.sh --out results/p5b3 --reuse /data/p5b_reuse --sections env correctness budget
       (/data/p5b_reuse: rates.json and sharegpt_sample.json of P5b, uploaded with the code; results/ isn't in git)
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = ["env", "correctness", "sharegpt", "spec", "mixed", "step_cost", "repeats"]
EXTRA_SECTIONS = ["budget"]    # P5b3, only when asked for

# sizes: full run on the rented GPU / --smoke rehearsal
FULL = dict(
    spec_num_seqs={1: 8, 4: 16, 16: 48, 64: 128}, spec_max_tokens=768,
    mixed_num_seqs=200, mixed_input=(1024, 2048), mixed_output=(256, 512),
    sharegpt_all=1000, sharegpt_rate=400,
    test_args=[], step_cost_args=[], common=[], timeout=3600,
)
SMOKE = dict(
    spec_num_seqs={1: 2, 4: 4, 16: 4, 64: 8}, spec_max_tokens=48,
    mixed_num_seqs=12, mixed_input=(256, 512), mixed_output=(32, 64),
    sharegpt_all=48, sharegpt_rate=24,
    test_args=["--max-tokens", "32", "--variants", "eager", "graph", "spec_ngram_graph", "spec_draft", "spec_draft_graph", "mixed_graph"],
    step_cost_args=["--lengths", "256", "--batch-sizes", "1", "4", "16", "--repeat", "5"],
    common=["--max-num-seqs", "64", "--gpu-memory-utilization", "0.8"], timeout=1200,
)
TEMPERATURES = (0, 0.6)
SPEC_CONFIGS = {
    "graph": [],
    "ngram_dynk": ["--spec", "ngram", "--k", "8", "--dynamic-k"],
    "draft_k4": ["--spec", "draft", "--k", "4", "DRAFT"],
    "draft_dynk": ["--spec", "draft", "--k", "8", "--dynamic-k", "DRAFT"],
}
SPEC_KEY_CELLS = [("copy", 0, 1), ("chat", 0, 1), ("copy", 0, 16)]    # (workload, temperature, concurrency) repeated
MIXED_CONFIGS = {
    "graph": [],
    "mixed512": ["--mixed", "--max-num-batched-tokens", "512"],
    "mixed2048": ["--mixed", "--max-num-batched-tokens", "2048"],
    "mixed512_ngram_dynk": ["--mixed", "--max-num-batched-tokens", "512", "--spec", "ngram", "--k", "8", "--dynamic-k"],
}
MIXED_KEY = ["graph", "mixed512"]
MIXED_LOADS = {"lo": 0.9, "hi": 1.5}    # Poisson rate as a multiple of the all-at-once request rate
SHAREGPT_CONFIGS = {    # (engine, extra bench.py arguments)
    "nano_plain": ("nanovllm", []),
    "nano_ngram": ("nanovllm", ["--spec", "ngram", "--k", "4"]),
    "nano_mixed512": ("nanovllm", ["--mixed", "--max-num-batched-tokens", "512"]),
    "vllm_plain": ("vllm", []),
    "vllm_ngram": ("vllm", ["--spec", "ngram", "--k", "4"]),
}
SHAREGPT_DRAFT = {"nano_draft": ("nanovllm", ["--spec", "draft", "--k", "4", "DRAFT"]),
                  "vllm_draft": ("vllm", ["--spec", "draft", "--k", "4", "DRAFT"])}
BUDGETS = [2048, 16384, 4096, 8192]    # most important first; the first two are BUDGET_KEY
BUDGET_KEY = [2048, 16384]
BUDGET_EXTRA = {"nano_plain": ("nanovllm", []),
                "nano_mixed2048_ngram_dynk": ("nanovllm", ["--mixed", "--max-num-batched-tokens", "2048", "--spec", "ngram", "--k", "8", "--dynamic-k"])}
# the eager/graph pairs of tests/test_greedy.py's MUST_MATCH
PAIR_VARIANTS = ["eager", "graph", "spec_ngram", "spec_ngram_graph", "spec_draft", "spec_draft_graph", "mixed_eager", "mixed_graph"]


class Runner:

    def __init__(self, args):
        self.args = args
        self.size = SMOKE if args.smoke else FULL
        self.out = args.out
        self.failures = []
        self.done = 0
        os.makedirs(self.out, exist_ok=True)
        self.baseline_mb = gpu_used_mb()

    def wait_for_gpu(self):
        """A process that just exited may still hold its GPU memory for a few seconds, and vLLM refuses to start
        when free memory is below gpu_memory_utilization: wait (at most 60s) until usage is back to the baseline."""
        for _ in range(60):
            if gpu_used_mb() <= self.baseline_mb + 256:
                return
            time.sleep(1)
        print(f"    GPU memory still at {gpu_used_mb()}MB (baseline {self.baseline_mb}MB), starting anyway", flush=True)

    def path(self, name):
        return os.path.join(self.out, f"{name}.json")

    def load(self, name):
        if not os.path.exists(self.path(name)):
            return None
        with open(self.path(name)) as f:
            return json.load(f)

    def run(self, name, cmd, kind="bench"):
        """Run cmd unless <name>.json exists. bench: bench.py writes the JSON itself (--out is appended);
        check: the JSON records whether the command succeeded."""
        if os.path.exists(self.path(name)):
            return self.load(name)
        if self.args.dry_run:
            print(f"[dry-run] {name}: {' '.join(cmd)}")
            return None
        print(f"[{datetime.datetime.now():%H:%M:%S}] {name}: {' '.join(cmd[1:])}", flush=True)
        self.wait_for_gpu()
        tmp = self.path(name) + ".part"
        full = cmd + (["--out", tmp] if kind == "bench" else [])
        start = time.time()
        log_path = os.path.join(self.out, f"{name}.log")
        with open(log_path, "w") as log:
            log.write(" ".join(full) + "\n\n")
            log.flush()
            try:
                rc = subprocess.run(full, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=self.size["timeout"]).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        wall = time.time() - start
        if kind == "bench":
            if rc != 0 or not os.path.exists(tmp):
                print(f"    FAILED ({rc}) after {wall:.0f}s, see {log_path}", flush=True)
                self.failures.append(name)
                return None
            with open(tmp) as f:
                result = json.load(f)
            os.remove(tmp)
        else:
            if rc == "timeout":
                print(f"    TIMEOUT after {wall:.0f}s, see {log_path}", flush=True)
                self.failures.append(name)
                return None
            result = {"ok": rc == 0, "returncode": rc}
            if rc != 0:
                print(f"    FAILED (exit {rc}), see {log_path}", flush=True)
                self.failures.append(name)
        result["wall_s"] = wall    # whole process: startup (loading, CUDA graphs, warmup) + the measurement
        result["cmd"] = full
        with open(self.path(name), "w") as f:
            json.dump(result, f, indent=2)
        self.done += 1
        print(f"    done in {wall:.0f}s", flush=True)
        return result

    def bench(self, name, args, engine="nanovllm"):
        python = self.args.vllm_python if engine == "vllm" else sys.executable
        args = [x for a in args for x in (["--draft-model", self.args.draft_model] if a == "DRAFT" else [a])]
        return self.run(name, [python, "bench.py", "--model", self.args.model, "--engine", engine] + self.size["common"] + args)

    # ---- sections

    def env(self):
        d = os.path.join(self.out, "env")
        if os.path.exists(os.path.join(d, "done")) or self.args.dry_run:
            return
        os.makedirs(d, exist_ok=True)
        cmds = {
            "nvidia-smi.txt": "nvidia-smi; nvidia-smi -q",
            "system.txt": "uname -a; cat /etc/os-release; lscpu; free -g; df -h; nproc",
            "git.txt": f"cd {ROOT} && git log -1 --format='%H %ad %s' && git status --short",
            "pip_freeze_nano.txt": f"{sys.executable} -m pip freeze",
            "pip_freeze_vllm.txt": f"{self.args.vllm_python} -m pip freeze" if self.args.vllm_python else "true",
            "versions.txt": f"{sys.executable} -c 'import sys, torch, flash_attn, transformers, triton; print(sys.version); "
                            f"print(\"torch\", torch.__version__, torch.version.cuda, \"flash_attn\", flash_attn.__version__, "
                            f"\"transformers\", transformers.__version__, \"triton\", triton.__version__); print(torch.cuda.get_device_name())'"
                            + (f"; {self.args.vllm_python} -c 'import vllm, torch; print(\"vllm\", vllm.__version__, \"torch\", torch.__version__, torch.version.cuda)'"
                               if self.args.vllm_python else ""),
        }
        for file, cmd in cmds.items():
            with open(os.path.join(d, file), "w") as f:
                subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, executable="/bin/bash")
        args = {k: v for k, v in vars(self.args).items()}
        with open(os.path.join(d, "run_p5_args.json"), "w") as f:
            json.dump(args, f, indent=2)
        open(os.path.join(d, "done"), "w").close()

    def correctness(self):
        self.run("correctness", [sys.executable, "tests/test_greedy.py", "--model", self.args.model,
                                 "--draft-model", self.args.draft_model] + self.size["test_args"], kind="check")
        if "budget" in self.args.sections:
            args = self.size["test_args"][:2] if self.args.smoke else []    # smoke: keep only --max-tokens 32
            self.run("correctness30", [sys.executable, "tests/test_greedy.py", "--model", self.args.model, "--draft-model",
                                       self.args.draft_model, "--num-prompts", "30", "--variants"] + PAIR_VARIANTS + args, kind="check")
            self.run("check_padding", [sys.executable, "scripts/check_padding.py"], kind="check")    # this GPU's kernels, in the log

    def sharegpt_sample(self):
        sample = os.path.join(self.out, "sharegpt_sample.json")
        if not os.path.exists(sample) and not self.args.dry_run:
            if not self.args.sharegpt or not os.path.exists(self.args.sharegpt):
                print("no ShareGPT dataset (--sharegpt) and no sharegpt_sample.json: skipping the ShareGPT runs", flush=True)
                return None
            subprocess.run([sys.executable, "scripts/sharegpt.py", "sample", "--dataset", self.args.sharegpt, "--model", self.args.model,
                            "--num", str(self.size["sharegpt_all"]), "--seed", "0", "--out", sample], cwd=ROOT, check=True)
        return sample

    def sharegpt(self, rep=1, only_repeated=False):
        sample = self.sharegpt_sample()
        if not sample:
            return
        all_args = ["--workload", "sharegpt", "--dataset", sample, "--temperature", "0"]
        n_all, n_rate = self.size["sharegpt_all"], self.size["sharegpt_rate"]
        self.bench(f"sharegpt_nano_plain_all_r{rep}", all_args + ["--num-seqs", str(n_all)])    # also the rate probe
        rates = self.rates("sharegpt", "sharegpt_nano_plain_all_r1")
        configs = dict(SHAREGPT_CONFIGS)
        if not only_repeated:
            configs.update(SHAREGPT_DRAFT)
        for config, (engine, extra) in configs.items():
            loads = ["all", "lo", "hi"] if config in SHAREGPT_CONFIGS else ["all", "lo"]
            for load in loads:
                name = f"sharegpt_{config}_{load}_r{rep}"
                if load == "all":
                    self.bench(name, all_args + ["--num-seqs", str(n_all)] + extra, engine)
                elif rates:
                    self.bench(name, all_args + ["--num-seqs", str(n_rate), "--rate", f"{rates[load]:.3f}"] + extra, engine)

    def rates(self, section, probe_name):
        """Poisson rates of a section: MIXED_LOADS times the request rate of its all-at-once probe run,
        fixed once (rates.json) so that repeats and resumed runs use the same."""
        path = os.path.join(self.out, "rates.json")
        rates = json.load(open(path)) if os.path.exists(path) else {}
        if section not in rates:
            probe = self.load(probe_name)
            if not probe:
                return None
            rates[section] = {load: f * probe["request_throughput"] for load, f in MIXED_LOADS.items()}
            rates[section]["capacity"] = probe["request_throughput"]
            with open(path, "w") as f:
                json.dump(rates, f, indent=2)
            print(f"{section}: all-at-once rate {probe['request_throughput']:.2f} requests/s -> Poisson rates "
                  + ", ".join(f"{k} {v:.2f}" for k, v in rates[section].items()), flush=True)
        return rates[section]

    def spec(self, rep=1, only_repeated=False):
        for workload in ("copy", "chat"):
            for t in TEMPERATURES:
                for c, n in self.size["spec_num_seqs"].items():
                    if only_repeated and (workload, t, c) not in SPEC_KEY_CELLS:
                        continue
                    for config, extra in SPEC_CONFIGS.items():
                        self.bench(f"spec_{workload}_T{t}_c{c}_{config}_r{rep}",
                                   ["--workload", workload, "--temperature", str(t), "--concurrency", str(c), "--num-seqs", str(n),
                                    "--max-tokens", str(self.size["spec_max_tokens"])] + extra)

    def mixed(self, rep=1, only_repeated=False):
        s = self.size
        base = ["--workload", "random", "--input-len", *map(str, s["mixed_input"]), "--output-len", *map(str, s["mixed_output"]),
                "--num-seqs", str(s["mixed_num_seqs"]), "--temperature", "0.6"]
        self.bench("mixed_probe_graph_all", base)
        rates = self.rates("mixed", "mixed_probe_graph_all")
        if not rates:
            return
        for load in MIXED_LOADS:
            for config, extra in MIXED_CONFIGS.items():
                if only_repeated and config not in MIXED_KEY:
                    continue
                self.bench(f"mixed_{config}_{load}_r{rep}", base + ["--rate", f"{rates[load]:.3f}"] + extra)

    def step_cost(self):
        for method in ("ngram", "draft"):
            extra = ["--draft-model", self.args.draft_model] if method == "draft" else []
            self.run(f"step_cost_{method}", [sys.executable, "scripts/step_cost.py", "--model", self.args.model, "--spec", method]
                     + extra + self.size["step_cost_args"], kind="check")

    def repeats(self):
        for rep in range(2, self.args.repeats + 1):
            self.sharegpt(rep, only_repeated=True)
            self.spec(rep, only_repeated=True)
            self.mixed(rep, only_repeated=True)

    def budget(self):
        sample = self.sharegpt_sample()
        if not sample:
            return
        all_args = ["--workload", "sharegpt", "--dataset", sample, "--temperature", "0"]
        n_all, n_rate = self.size["sharegpt_all"], self.size["sharegpt_rate"]
        rates = self.rates("sharegpt", "sharegpt_nano_plain_all_r1")
        if not rates:    # no --reuse: measure the rates here, as P5b did
            self.bench("sharegpt_nano_plain_all_r1", all_args + ["--num-seqs", str(n_all)])
            rates = self.rates("sharegpt", "sharegpt_nano_plain_all_r1")
        for rep in range(1, self.args.repeats + 1):
            configs = {}
            for b in BUDGETS if rep == 1 else BUDGET_KEY:
                configs[f"nano_mixed{b}"] = ("nanovllm", ["--mixed", "--max-num-batched-tokens", str(b)])
                if self.args.vllm_python:
                    configs[f"vllm_b{b}"] = ("vllm", ["--max-num-batched-tokens", str(b)])
            if rep == 1:
                configs.update(BUDGET_EXTRA)
            for config, (engine, extra) in configs.items():
                for load in ("all", "lo", "hi") if rates else ("all",):
                    load_args = ["--num-seqs", str(n_all)] if load == "all" else ["--num-seqs", str(n_rate), "--rate", f"{rates[load]:.3f}"]
                    self.bench(f"budget_{config}_{load}_r{rep}", all_args + load_args + extra, engine)
            if rep == 1:
                # where a mixed step's time goes on this GPU (P5b: a <= 512-token mixed step took 82ms, a 512-token prefill 42ms)
                for b in (512, 2048):
                    self.run(f"profile_mixed_{b}", [sys.executable, "scripts/profile_mixed.py", "--model", self.args.model, "--budget", str(b),
                                                    "--num-seqs", "12" if self.args.smoke else "48"], kind="check")

    def package(self):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        tar_path = os.path.join(os.path.dirname(os.path.abspath(self.out)), f"{os.path.basename(os.path.abspath(self.out))}_{stamp}.tar.gz")
        setup_log = os.path.join(os.environ.get("P5", ""), "setup.log")
        if os.path.exists(setup_log):
            shutil.copy(setup_log, os.path.join(self.out, "env", "setup.log"))
        with open(os.path.join(self.out, "tables.md"), "w") as f:
            print_tables(self.out, f)
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(self.out, arcname=os.path.basename(os.path.abspath(self.out)))
        print(f"packed {tar_path} ({os.path.getsize(tar_path) / 1e6:.1f}MB)")
        return tar_path


def gpu_used_mb():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(r.stdout.split()[0]) if r.returncode == 0 else 0


# ---- tables: every cell as mean [min..max] over its repeats

def collect(out):
    """{cell name (without _rN): [result, ...]}"""
    cells = {}
    for file in sorted(os.listdir(out)):
        if not file.endswith(".json") or file in ("rates.json", "sharegpt_sample.json"):
            continue
        name = file[:-5]
        with open(os.path.join(out, file)) as f:
            r = json.load(f)
        base, _, rep = name.rpartition("_r")
        cells.setdefault(base if rep.isdigit() else name, []).append(r)
    return cells


def stat(rs, get, fmt="{:.1f}"):
    vals = [get(r) for r in rs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    mean = sum(vals) / len(vals)
    return fmt.format(mean) + (f" [{fmt.format(min(vals))}..{fmt.format(max(vals))}]" if len(vals) > 1 else "")


def print_tables(out, f=sys.stdout):
    cells = collect(out)
    w = lambda line="": print(line, file=f)
    w("| cell | runs | output tok/s | TTFT P50 | TTFT P99 | TPOT mean | ITL P50 | ITL P99 | ITL max | acceptance | tokens per arrival | KV tokens | wall s |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rs in cells.items():
        if "output_throughput" not in rs[0]:
            continue
        spec = lambda key: (lambda r: r.get("spec_stats", {}).get(key))
        w(f"| {name} | {len(rs)} | {stat(rs, lambda r: r['output_throughput'], '{:.0f}')} | {stat(rs, lambda r: r['ttft_ms']['p50'], '{:.0f}')} | "
          f"{stat(rs, lambda r: r['ttft_ms']['p99'], '{:.0f}')} | {stat(rs, lambda r: r['tpot_ms']['mean'])} | {stat(rs, lambda r: r['itl_ms']['p50'])} | "
          f"{stat(rs, lambda r: r['itl_ms']['p99'])} | {stat(rs, lambda r: r['itl_ms']['max'], '{:.0f}')} | {stat(rs, spec('acceptance_rate'), '{:.2f}')} | "
          f"{stat(rs, lambda r: r.get('tokens_per_arrival'), '{:.2f}')} | {rs[0].get('kv_tokens', '-')} | {stat(rs, lambda r: r.get('wall_s'), '{:.0f}')} |")
    w()
    for name, rs in cells.items():
        if "ok" in rs[0]:
            w(f"- {name}: {'PASS' if rs[0]['ok'] else 'FAIL'} ({rs[0]['wall_s']:.0f}s)")
    rates = os.path.join(out, "rates.json")
    if os.path.exists(rates):
        w(f"- Poisson rates (requests/s): {open(rates).read()}")
    wall = sum(r.get("wall_s", 0) for rs in cells.values() for r in rs)
    w(f"- total wall time of all runs: {wall / 3600:.2f} h")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL"))
    p.add_argument("--draft-model", default=os.environ.get("DRAFT_MODEL"))
    p.add_argument("--vllm-python", default=os.environ.get("VLLM_PY"), help="python of the venv with vLLM (none: skip the vLLM runs)")
    p.add_argument("--sharegpt", default=os.environ.get("SHAREGPT"), help="ShareGPT_V3_unfiltered_cleaned_split.json")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "p5"))
    p.add_argument("--smoke", action="store_true", help="tiny runs, to rehearse the whole script on a small GPU")
    p.add_argument("--sections", nargs="+", choices=SECTIONS + EXTRA_SECTIONS, default=SECTIONS)
    p.add_argument("--reuse", help="earlier result folder: copy its rates.json and sharegpt_sample.json (same rates, same requests)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="print the runs that are still to do")
    p.add_argument("--table-only", action="store_true")
    p.add_argument("--no-package", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="run the performance sections even if the correctness test failed")
    args = p.parse_args()
    if args.table_only:
        print_tables(args.out)
        return
    if not args.model or not args.draft_model:
        p.error("--model and --draft-model are needed (or MODEL / DRAFT_MODEL, see p5.env)")
    if not args.vllm_python:
        print("no --vllm-python: the vLLM runs are skipped", flush=True)
        for key in ("vllm_plain", "vllm_ngram"):
            SHAREGPT_CONFIGS.pop(key)
        SHAREGPT_DRAFT.pop("vllm_draft")

    runner = Runner(args)
    if args.reuse:
        for file in ("rates.json", "sharegpt_sample.json"):
            if not os.path.exists(os.path.join(args.out, file)):
                shutil.copy(os.path.join(args.reuse, file), os.path.join(args.out, file))
    start = time.time()
    for section in args.sections:
        print(f"\n===== {section} ({datetime.datetime.now():%H:%M:%S})", flush=True)
        getattr(runner, section)()
        check = runner.load("correctness")
        if section == "correctness" and check is not None and not check["ok"] and not args.keep_going:
            # numbers from an engine that gives wrong outputs are worthless: don't spend GPU hours on them
            print("correctness test FAILED (see correctness.log): stopping before the performance runs. "
                  "Fix it, delete correctness.json and rerun, or pass --keep-going to measure anyway.", flush=True)
            sys.exit(1)
    print(f"\n{runner.done} runs done in {(time.time() - start) / 3600:.2f} h this time", flush=True)
    if runner.failures:
        print(f"FAILED: {' '.join(runner.failures)} (rerun the script to retry the ones without a JSON)")
    if not args.dry_run and not args.no_package:
        runner.package()
    print_tables(args.out)
    sys.exit(1 if runner.failures else 0)


if __name__ == "__main__":
    main()
