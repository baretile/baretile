# Four benchmarks, run on baretile or on TT-NN on the same card. Each run records the versions that took part
# (baretile, tt-metal and the UMD it pins, tt-kmd, the card's firmware, the host) and the lines of code that cloc
# counts. It prints the results and writes them to benchmark-<stack>.json.
#   B1, matmul: (4096 x 4096) @ (4096 x 4096) in bf16 with fp32 sums, at 2 and at 4 fidelity phases. After 3 warm-up
#     runs, it reports the median of 5 measurements. Each is 20 runs back to back, from the first launch to the last
#     result.
#   B2, decode: Llama-3.1-8B-Instruct at TT's performance precision, for 1 user and for 32. It runs 200 greedy steps
#     after the reference file's first 512 tokens, timed once the stack's step replays. It also reports the effective
#     DRAM bandwidth. A token reads 6.20 GB, so 41.3 tokens/s is half the card's 512 GB/s. Then it measures
#     Tenstorrent's token accuracy over 500 steps. Each step is fed the file's next token and scored top-1 and top-5
#     against the reference model.
#   B3, long prefill: time to first token for B2's model with prompts of 4,096 and 32,768 tokens, after a warm-up.
#   B4, serving: B2's model behind serve.py, with 32 concurrent agents over HTTP. Each agent runs four turns. They
#     start from a shared system prompt, and each turn adds the last answer and a tool result. It reports tokens/s per
#     user and for the card (including first-token latency), time to first text, and the share of prompt tokens the server had cached. B4 has no
#     TT-NN side, since TT-NN serves through vLLM, but `load` can drive any OpenAI-compatible server.
# `table` compares both stacks. It reports the speed ratio (baretile / TT-NN; above 1 means baretile is faster)
# and the fraction of the card's theoretical limit. That limit is the peak matmul rate at the benchmark's
# fidelity for B1 and B3, and the DRAM bandwidth for B2.
#   python3 test/benchmark.py baretile path/to/Llama-3.1-8B-Instruct path/to/Llama-3.1-8B-Instruct.refpt [--quick]
#   python3 test/benchmark.py ttnn path/to/Llama-3.1-8B-Instruct path/to/Llama-3.1-8B-Instruct.refpt [--quick]
#   python3 test/benchmark.py table benchmark-baretile.json benchmark-ttnn.json   # the two side by side, as markdown
#   python3 test/benchmark.py load http://localhost:8000 [agents]   # B4's load against a server already running
# The folder holds the model's Hugging Face files. The reference file is tt_transformers'
# tests/reference_outputs/<model>.refpt. B3's prompts are the opening of A Tale of Two Cities, which tt_transformers
# keeps in tests/tale-of-two-cities.txt.bz2. Set BOOK=path to use another copy.
# ttnn runs in tt-metal's Python environment, with TT_METAL_HOME set to its checkout. B1 is timed the way TT-NN's
# tutorial times a matmul. B2 and B3 come from Tenstorrent's demo script (simple_text_demo.py, the performance runs
# named below) with our prompts and step counts, and its logs go to benchmark-ttnn-<run>.log. TT_KMD=path/to/tt-kmd
# counts tt-kmd's lines too. On ttsim (TTSIM for baretile, TT_METAL_SIMULATOR for TT-NN) the times are ttsim's clock.
# --quick shrinks the sizes, cuts the model to two blocks and decodes after the last 32 tokens of the prompt, which is
# enough to test the script.
# TT-NN's B2 and B3 don't run on ttsim, because Tenstorrent's script warms up prefill with an SFPU mode that ttsim
# lacks.
import bz2, glob, http.client, json, math, os, platform, re, shutil, signal, socket, statistics, subprocess, sys, tempfile, threading, time
from urllib.parse import urlsplit
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile import bh, pci
from models import cost, gpt2, llama

ROOT = os.path.join(os.path.dirname(__file__), "..")
SIZES = {False: SimpleNamespace(n=4096, runs=(3, 5, 20), steps=200, accuracy=500, users=32, lengths=(4096, 32768), blocks=None,
                               turns=SimpleNamespace(system=1500, task=400, tool=200, out=256, n=4)),  # B4's prompt words, answer tokens and turns
         True: SimpleNamespace(n=256, runs=(1, 2, 2), steps=8, accuracy=8, users=2, lengths=(64, 128), blocks=2,
                               turns=SimpleNamespace(system=24, task=6, tool=6, out=8, n=2))}

def git(folder):  # a checkout's commit, and whether it has changes
  run = lambda *a: subprocess.run(["git", "-C", folder, *a], capture_output=True, text=True, timeout=60).stdout.strip()
  return run("rev-parse", "--short", "HEAD") + (" with changes" if run("status", "--porcelain", "--untracked-files=no") else "")

def lines(folder, keep):  # lines of code that cloc counts in the git-tracked files that keep accepts, submodules included
  if not shutil.which("cloc"): return "not counted: cloc is not installed"
  files = subprocess.run(["git", "-C", folder, "ls-files", "--recurse-submodules"], capture_output=True, text=True, timeout=60).stdout.splitlines()
  with tempfile.NamedTemporaryFile("w") as f:
    f.write("\n".join(os.path.join(folder, p) for p in files if keep(p)))
    f.flush()
    out = subprocess.run(["cloc", "--json", "--quiet", f"--list-file={f.name}"], capture_output=True, text=True, timeout=60).stdout
  try: return json.loads(out)["SUM"]["code"]
  except (ValueError, KeyError): return f"not counted: cloc found no code in {folder}"

def tenstorrent(path):
  # Tenstorrent's code as we count it. We leave out tests, models, docs, Markdown, CI and editor settings (dot folders),
  # and third-party code except the UMD that tt-metal pins.
  parts = path.split("/")
  return (not {"tests", "test", "models", "docs"} & set(parts) and not any(p.startswith(".") for p in parts) and not path.endswith(".md")
          and all(parts[i + 1] == "umd" for i, p in enumerate(parts[:-1]) if p == "third_party"))

def host():
  first = lambda path, key: next((l.split(key, 1)[1].strip().strip('"') for l in open(path) if l.startswith(key)), None)
  return {"cpu": first("/proc/cpuinfo", "model name\t:"), "os": first("/etc/os-release", "PRETTY_NAME="), "kernel": platform.release(),
          "python": platform.python_version(), "numpy": np.__version__}

def card(sim):  # which card, and its PCIe link
  if sim: return "ttsim"
  bdf = pci.find_cards()[0]
  read = lambda f: open(f"/sys/bus/pci/devices/{bdf}/{f}").read().strip()
  return f"{bdf}, PCIe {read('current_link_speed')} x{read('current_link_width')}"

def timed(run, wait, clock, runs):
  # seconds per run, the median of runs[1] measurements of runs[2] runs back to back, after runs[0] warm-up runs
  for _ in range(runs[0]): run()
  wait()
  times = []
  for _ in range(runs[1]):
    start = clock()
    for _ in range(runs[2]): run()
    wait()
    times.append((clock() - start) / runs[2])
  return statistics.median(times)

def load(url, agents, turns, text, timeout=3600):
  """Run streamed greedy completions, with one deadline for the whole workload.

  Each agent extends its shared system prompt with a task, then each answer and tool result. Rates use reported
  token counts and complete request durations, including the first token. SSE text chunks are not token timestamps;
  the first-text latency is all the wire measures. Missing usage or cache counts produce None. On failure or timeout,
  completed turns remain in the report, with failures and the total elapsed time (including unfinished requests)."""
  if not isinstance(agents, int) or agents < 1 or not isinstance(turns.n, int) or turns.n < 1:
    raise ValueError("benchmark agents and turns must be positive integers")
  if not math.isfinite(timeout) or timeout <= 0: raise ValueError("benchmark timeout must be positive and finite")
  endpoint = urlsplit(url)
  if endpoint.scheme not in ("http", "https") or not endpoint.hostname: raise ValueError("benchmark URL must use HTTP or HTTPS")
  path = endpoint.path.rstrip("/") + "/v1/completions" + ("?" + endpoint.query if endpoint.query else "")
  words, done, failed, sockets = text.split(), [], {}, {}
  lock, stopped = threading.Lock(), threading.Event()
  beginning = time.perf_counter()
  deadline = beginning + timeout
  piece = lambda at, n: " " + " ".join(words[at:at + n])
  def remaining():
    left = deadline - time.perf_counter()
    if stopped.is_set() or left <= 0: raise TimeoutError("benchmark workload deadline exceeded")
    return left
  def agent(a):
    at = turns.system + a * (turns.task + turns.n * turns.tool)
    prompt = piece(0, turns.system) + piece(at, turns.task)
    try:
      for k in range(turns.n):
        body = json.dumps({"prompt": prompt, "max_tokens": turns.out, "temperature": 0, "stream": True,
                           "stream_options": {"include_usage": True}}).encode()
        start, first, answer, usage = time.perf_counter(), None, "", None
        cls = http.client.HTTPSConnection if endpoint.scheme == "https" else http.client.HTTPConnection
        conn = cls(endpoint.hostname, endpoint.port, timeout=remaining())
        try:
          conn.connect()
          with lock:
            remaining()
            sockets[a] = conn.sock  # keep the socket even when getresponse gives its ownership to the response
          conn.request("POST", path, body, {"Content-Type": "application/json"})
          with conn.getresponse() as response:
            if response.status != 200: raise RuntimeError(f"HTTP {response.status}: {response.reason}")
            while True:
              remaining()
              line = response.readline(1 << 20)
              if not line: break
              if len(line) == 1 << 20: raise ValueError("benchmark SSE line exceeds 1 MiB")
              if not line.startswith(b"data:"): continue
              data = line[5:].strip()
              if data == b"[DONE]": break
              chunk = json.loads(data)
              if "error" in chunk: raise RuntimeError(str(chunk["error"]))
              usage = chunk.get("usage") or usage
              choices = chunk.get("choices") or []
              if choices and (part := choices[0].get("text")):
                if first is None: first = time.perf_counter()
                answer += part
          end = time.perf_counter()
          with lock:
            remaining()
            done.append((start, first, end, usage))
        finally:
          with lock: sockets.pop(a, None)
          conn.close()
        prompt += answer + piece(at + turns.task + k * turns.tool, turns.tool)
    except Exception as e:
      with lock: failed.setdefault(a, f"agent {a}: {type(e).__name__}: {e}")
  threads = [threading.Thread(target=agent, args=(a,), daemon=True) for a in range(agents)]
  for t in threads: t.start()
  for t in threads: t.join(max(0, deadline - time.perf_counter()))
  with lock:
    stopped.set()
    for a, t in enumerate(threads):
      if t.is_alive(): failed.setdefault(a, f"agent {a}: TimeoutError: benchmark workload deadline exceeded")
    for sock in sockets.values():
      try: sock.shutdown(socket.SHUT_RDWR)
      except OSError: pass
    completed, errors = list(done), list(failed.values())
  # Socket shutdown interrupts header and body reads, including streams that keep sending a byte at a time.
  cleanup = time.perf_counter() + 1
  for t in threads: t.join(max(0, cleanup - time.perf_counter()))
  elapsed = time.perf_counter() - beginning
  first = sorted(first - start for start, first, _, _ in completed if first is not None)
  count = lambda u, key: u.get(key) if isinstance(u, dict) and type(u.get(key)) is int and u[key] >= 0 else None
  totals = [count(u, "completion_tokens") for *_, u in completed]
  prompts = [count(u, "prompt_tokens") for *_, u in completed]
  cached = [count(u.get("prompt_tokens_details"), "cached_tokens") if isinstance(u, dict) else None for *_, u in completed]
  known = bool(completed) and all(n is not None for n in totals)
  each = [n / (end - start) for (start, _, end, _), n in zip(completed, totals) if n is not None and end > start]
  cache = sum(cached) / sum(prompts) if completed and all(n is not None for n in prompts + cached) and sum(prompts) else None
  return {"agents": agents, "turns": len(completed), "seconds": elapsed, "failures": errors,
          "tokens/s a user": statistics.median(each) if known and each else None, "tokens/s": sum(totals) / elapsed if known else None,
          "seconds to first text": statistics.median(first) if first else None,
          "seconds to first text, 90th percentile": first[int(0.9 * (len(first) - 1))] if first else None, "prompt tokens cached": cache}

def logged(args, path, timeout=3600, **kwargs):
  """Run a benchmark subprocess to a log, stopping its process group at the deadline and keeping partial output."""
  if not math.isfinite(timeout) or timeout <= 0: raise ValueError("benchmark timeout must be positive and finite")
  with open(path, "w") as log:
    process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, **kwargs)
    try: return process.wait(timeout), False
    except subprocess.TimeoutExpired:
      try: os.killpg(process.pid, signal.SIGTERM)
      except ProcessLookupError: pass
      try: process.wait(min(5, timeout))
      except subprocess.TimeoutExpired:
        pass
      finally:
        # A subprocess can exit before its children. Stop the whole group even when its leader has already exited.
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(5)
      log.write(f"\nBenchmark deadline exceeded after {timeout:g} seconds.\n")
      return process.returncode, True

def weights(c):  # rows, columns and format of each weight as llama.upload makes it, from the model's sizes alone
  hd = llama.head(c)
  qkv = (c.heads + 2 * c.kv) * hd
  shapes = {"embed": (c.vocab, c.dim), "embed.T": (c.dim, c.vocab), "norm": (1, c.dim), "cos": (c.places, qkv), "sin": (c.places, qkv),
            "turn": (hd, hd)}
  block = {"input_layernorm": (1, c.dim), "post_attention_layernorm": (1, c.dim), "qkv_proj": (c.dim, qkv), "o_proj": (c.heads * hd, c.dim),
           "gate_up_proj": (c.dim, 2 * c.ffn), "down_proj": (c.ffn, c.dim)}
  for l in range(c.layers): shapes |= {f"layers.{l}.{k}": s for k, s in block.items()}
  return {k: SimpleNamespace(rows=r, cols=n, dtype=llama.weight_format(k, c.precision)) for k, (r, n) in shapes.items()}

def bytes_per_token(t, c, context):
  """Bytes that a decode token reads in B2. It reads the weights at their stored sizes, plus the KV cache at the
  decode's mean context. From the embedding and RoPE tables it reads one row each."""
  rows = {"embed": c.dim, "cos": t["cos"].cols, "sin": t["sin"].cols}
  return (sum(cost.BYTES[x.dtype] * rows.get(k, x.rows * x.cols) for k, x in t.items())
          + context * c.layers * 2 * c.kv * llama.head(c) * cost.BYTES[llama.cache_format(c)])

def roofline(flops, seconds, phases):  # a benchmark's share of the card's peak matmul rate at its fidelity phases
  return flops / seconds / (cost.CARD["tflops"] / phases)

def prefill_flops(c, n, phases=False):
  # B3's matmul flops, or phase-weighted work when phases=True. Each token goes through the weights, but only the last goes through the output layer.
  # Attention adds its scores and values over the rows that each token sees.
  s = cost.shape(c, "tt")
  values, head = (s.work, s.head_work) if phases else (s.values, c.vocab * c.dim)
  attention = c.layers * 4 * c.heads * llama.head(c) * n * (n + 1) // 2
  return 2 * n * (values - head) + 2 * head + attention * ((s.prefill_phases if n > 1 else s.decode_phases) if phases else 1)

def model(folder, s, places):  # the model's config at TT's performance precision, and its weights, read lazily
  c, w = llama.load(folder, places)
  if s.blocks: c.layers = s.blocks
  c.precision = llama.precision(c, "tt")
  return c, {k: v for k, v in w.items() if not k.startswith("layers.") or int(k.split(".")[1]) < c.layers}

def prompts(prompt, s):
  """The prompt that B2 decodes after, and the prompt its token accuracy is scored from. --quick decodes after the last
  32 tokens to save time, but accuracy keeps the reference's prompt, because the reference's top five came from it."""
  return (prompt[-32:] if s.blocks else prompt), prompt

def baretile(folder, prompt, after, top5, book, s):
  short, prompt = prompts(prompt, s)
  from baretile.chip import Chip
  from baretile.runtime import Runtime
  from baretile.tensor import Tensor, jit
  chip = Chip(pci.open_device())
  rt, clock, on, sim = Runtime(chip), chip.pci.now, lambda row: Tensor.of(rt, [row], "int32"), isinstance(chip.pci, pci.Sim)
  tables = lambda n: llama.upload(rt, {k: w for k, w in llama.rope_weights(c, n).items() if k != "turn"})  # RoPE tables for n positions
  firmware = chip.telemetry().get(bh.TAG_FIRMWARE, 0)
  code = {d: lines(ROOT, lambda p: p.startswith(d + "/")) for d in ("baretile", "adapters")}
  versions = {"baretile": git(ROOT), "firmware": ".".join(str(firmware >> b & 255) for b in (24, 16, 8)), "card": card(sim),
              "AI clock MHz": chip.aiclk, "host": host(), "lines of code": code}
  out = {"versions": versions, "clock": "ttsim's, so this only checks the script" if sim else "the host's"}

  def decode(t, c, users):
    """tokens/s per user, with `users` decoding greedily together after the same prompt, ignoring end-of-sequence.
    The first two steps are warm-up (jit runs, then records). Each step is timed from the token ids on the host to the
    next ones there."""
    n, size = len(short), -(-(len(short) + s.steps + 1) // 32) * 32  # a slot's last row is for idle slots
    kv, tokens = llama.kv_cache(c, size * users, llama.zeros(rt, llama.cache_format(c)), users), []
    for u in range(users):  # prefill each user's prompt into its part of the cache, and get its next token
      tokens.append(int(llama.logits(llama.forward(on(short), on(list(range(n))), t, c, llama.slot(c, kv, u, users))).argmax()))
    step, times = jit(lambda ids, places: llama.forward(ids, places, t, c, kv, users if users > 1 else None)), []
    for i in range(s.steps):
      start = clock()
      ids, places = on(tokens), on([n + i] * users)
      tokens = [int(x) for x in gpt2.values(step(ids, places)).argmax(-1)]
      times.append(clock() - start)
    return (s.steps - 2) / sum(times[2:])

  def prefill(t, c, ids):
    """Seconds from ids on the host to the first token, after a warm-up prefill, and the chunk size llama.prefill used.
    The RoPE tables and the cache are sized for these ids. A chunk is (1 << 31) // (2 * heads * cache rows) tokens,
    at least 32: a bound from when attention's scores went to DRAM. Tensor.attend keeps them in L1 now, so the chunk
    size is for the card to tune."""
    t, size = t | tables(len(ids)), -(-len(ids) // 32) * 32
    kv, chunk = llama.kv_cache(c, size, llama.zeros(rt, llama.cache_format(c))), min(len(ids), max(32, (1 << 31) // (2 * c.heads * size) // 32 * 32))
    for _ in range(2):
      start = clock()
      int(llama.logits(llama.prefill(ids, t, c, kv, chunk, on)).argmax())
    return clock() - start, chunk

  def matmul(phases):  # B1. Its tensors free their DRAM before the model is uploaded
    rng = np.random.default_rng(0)
    a, b = (Tensor.frombytes(rt, s.n, s.n, (rng.standard_normal((s.n, s.n), np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes())
            for _ in range(2))
    seconds = timed(lambda: a.matmul(b, fidelity=phases).realize(), rt.wait, clock, s.runs)
    return {"ms": seconds * 1e3, "TFLOPS": 2 * s.n**3 / seconds / 1e12, "roofline": roofline(2 * s.n**3, seconds, phases)}
  for phases in (2, 4): out[f"B1 matmul {s.n}^3 bf16, {phases} phases"] = matmul(phases)
  c, w = model(folder, s, len(prompt) + max(s.steps, s.accuracy))
  t = llama.upload(rt, w, c.precision)
  tps = decode(t, c, 1)
  read = bytes_per_token(t, c, len(short) + (s.steps + 1) / 2)
  out["B2 decode, batch 1"] = {"tokens/s": tps, "GB a token": read / 1e9, "effective DRAM bandwidth": tps * read / (bh.DRAM_GBPS * 1e9)}
  tps = decode(t, c, s.users)
  out[f"B2 decode, batch {s.users}"] = {"tokens/s a user": tps, "tokens/s": tps * s.users}
  top1, five, _ = llama.accuracy(rt, t, c, prompt, after, top5, s.accuracy)
  out["B2 token accuracy"] = {"top-1": top1, "top-5": five, "steps": s.accuracy}
  ids = llama.Tokenizer(folder).encode(book)
  for n in s.lengths:  # a prompt that baretile can't run is reported as such
    try:
      seconds, chunk = prefill(t, c, ids[:n])
      out[f"B3 prefill, {n} tokens"] = {"seconds to first token": seconds, "chunk": chunk,
                                        "roofline": roofline(prefill_flops(c, n, phases=True), seconds, 1)}
    except Exception as e: out[f"B3 prefill, {n} tokens"] = {"does not run": f"{type(e).__name__}: {e}"}
  from models import serve  # B4: a slot for each agent, with room for two tokens a word plus the answers
  places = -(-(2 * (s.turns.system + s.turns.task + s.turns.n * s.turns.tool) + s.turns.n * s.turns.out) // 32) * 32
  engine = serve.Engine(rt, t | tables(places), c, llama.Tokenizer(folder), s.users, places)
  threading.Thread(target=engine.run, daemon=True).start()
  httpd = serve.server(engine, os.path.basename(folder.rstrip("/")), port=0)
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  out[f"B4 serving, {s.users} agents"] = load(f"http://localhost:{httpd.server_address[1]}", s.users, s.turns, book)
  httpd.shutdown()
  return out

DEMO = "models/tt_transformers/demo/simple_text_demo.py"
LOGGED = {"tokens/s a user": (r"@ ([\d.]+) tok/s/user", 1), "tokens/s": (r"\(([\d.]+) tok/s throughput\)", 1),
          "seconds to first token": (r"Time to First Token \(TTFT\): ([\d.]+)ms", 1e-3), "prompt tokens": (r"Encoded prompt lengths:\s*\[?(\d+)", 1),
          "top-1": (r"Top1 Accuracy: ([\d.]+)%", 1e-2), "top-5": (r"Top5 Accuracy: ([\d.]+)%", 1e-2)}

def ttnn(folder, prompt, after, top5, book, s):
  short = prompts(prompt, s)[0]  # TT's accuracy run reads its own reference file, so this prompt only goes to B2's decode
  import torch
  from transformers import AutoTokenizer
  tt, kmd, sim = os.environ["TT_METAL_HOME"], os.environ.get("TT_KMD"), "TT_METAL_SIMULATOR" in os.environ
  read = lambda path: open(path).read().strip() if os.path.exists(path) else None
  out = {"versions": {"tt-metal": git(tt), "umd": git(f"{tt}/tt_metal/third_party/umd"), "tt-kmd": read("/sys/module/tenstorrent/version"),
                      "firmware": next(map(read, glob.glob("/sys/class/tenstorrent/*/tt_fw_bundle_ver")), None), "card": card(sim),
                      "torch": torch.__version__, "host": host(),
                      "lines of code": {"tt-metal": lines(tt, tenstorrent),
                                        "tt-kmd": lines(kmd, tenstorrent) if kmd else "not counted: TT_KMD not set"}},
         "clock": "ttsim's, so this only checks the script" if sim else "the host's"}

  def matmul(n, phases):
    """Seconds per matmul, called and timed the way TT-NN's tutorial does (ttnn/tutorials/ttnn_intro.ipynb). Sums are
    fp32 (fp32_dest_acc_en, packer_l1_acc), and the inputs and result are bf16 in DRAM."""
    import ttnn
    device = ttnn.open_device(device_id=0)
    try:
      torch.manual_seed(0)
      upload = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
      a, b = (upload(torch.randn((n, n), dtype=torch.bfloat16)) for _ in range(2))
      config = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, f"HiFi{phases}"), math_approx_mode=False,
                                                fp32_dest_acc_en=True, packer_l1_acc=True)
      return timed(lambda: ttnn.matmul(a, b, compute_kernel_config=config), lambda: ttnn.synchronize_device(device), time.monotonic, s.runs)
    finally: ttnn.close_device(device)

  def demo(label, run, want, *flags, text=None):
    """Runs `run` from Tenstorrent's script at TT's performance precision, with flags overriding its settings and text
    as its prompt. Returns the values of `want` it logs, and why it failed if it did."""
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
      if text is not None: json.dump([{"prompt": text}], f); f.flush(); flags += ("--input_prompts", f.name, "--instruct", "0")
      if s.blocks: flags += ("--num_layers", str(s.blocks))
      path = f"benchmark-ttnn-{label}.log"
      code, expired = logged([sys.executable, "-m", "pytest", DEMO, "-k", f"performance and {run}", "-s", "-p", "no:cacheprovider", *flags],
                             path, timeout=3600, cwd=tt, env=os.environ | {"HF_MODEL": folder})
    log = open(path).read()
    got = {k: float(m[-1]) * LOGGED[k][1] for k in want + ("prompt tokens",) if (m := re.findall(LOGGED[k][0], log))}
    if expired: got["did not pass"] = ["benchmark deadline exceeded after 3600 seconds; partial output is in " + path]
    elif code or set(want) - set(got):
      got["did not pass"] = [l for l in log.splitlines() if l.startswith(("E ", "FAILED", "SKIPPED")) or "ERROR:" in l][-3:] or log.splitlines()[-1:]
    return got

  for phases in (2, 4):
    seconds = matmul(s.n, phases)
    out[f"B1 matmul {s.n}^3 bf16, {phases} phases"] = {"ms": seconds * 1e3, "TFLOPS": 2 * s.n**3 / seconds / 1e12,
                                                       "roofline": roofline(2 * s.n**3, seconds, phases)}
  tok = AutoTokenizer.from_pretrained(folder)
  text = tok.decode(short)  # TT's token accuracy makes its prompt from the reference tokens this way (prepare_ref_tokens)
  decoding = ("--stop_at_eos", "0", "--max_generated_tokens", str(s.steps))
  out["B2 decode, batch 1"] = demo("b2-batch-1", "batch-1", ("tokens/s",), *decoding, text=text)
  c, _ = model(folder, s, 1)
  if "tokens/s" in out["B2 decode, batch 1"]:
    read = bytes_per_token(weights(c), c, len(short) + (s.steps + 1) / 2)
    b2 = out["B2 decode, batch 1"]
    b2 |= {"GB a token": read / 1e9, "effective DRAM bandwidth": b2["tokens/s"] * read / (bh.DRAM_GBPS * 1e9)}
  out[f"B2 decode, batch {s.users}"] = demo(f"b2-batch-{s.users}", "batch-32 and not log-probs", ("tokens/s a user", "tokens/s"), *decoding,
                                            *(("--batch_size", str(s.users)) if s.users != 32 else ()), text=text)
  out["B2 token accuracy"] = demo("b2-accuracy", "ci-token-matching", ("top-1", "top-5"), "--max_generated_tokens", str(s.accuracy))
  ids = tok.encode(book, add_special_tokens=False)
  for n in s.lengths:  # the script adds a beginning-of-text token to n - 1 tokens of the book. --max_seq_len leaves room for 2n
    b3 = out[f"B3 prefill, {n} tokens"] = demo(f"b3-{n}", "long-context-64k", ("seconds to first token",), "--mode", "prefill",
                                               "--max_seq_len", str(2 * n), "--max_generated_tokens", "1", text=tok.decode(ids[:n - 1]))
    if "seconds to first token" in b3: b3["roofline"] = roofline(prefill_flops(c, n, phases=True), b3["seconds to first token"], 1)
  return out

def table(baretile, ttnn):
  """Return a Markdown table comparing both stacks and their shares of the hardware limits.
  Report baretile / TT-NN for rates and TT-NN / baretile for times, so a speed ratio above 1 always
  means baretile is faster."""
  show = lambda v: ", ".join(f"{k} {x:.4g}" if isinstance(x, float) else f"{k} {x}" for k, x in v.items()) if isinstance(v, dict) else str(v)
  rows = ["| | baretile | TT-NN | speedup over TT-NN |", "|---|---|---|---|"]
  speeds = {"B1": ("ms", -1), "B2 decode": ("tokens/s", 1), "B3": ("seconds to first token", -1)}  # -1: a time, so the ratio is inverted
  for key in [k for k in {**baretile, **ttnn} if k.startswith("B")]:
    f, t = baretile.get(key, {}), ttnn.get(key, {})
    speed = next((v for p, v in speeds.items() if key.startswith(p)), None)
    ratio = (f[speed[0]] / t[speed[0]]) ** speed[1] if speed and speed[0] in f and speed[0] in t else None
    gate = "" if ratio is None else f"{ratio:.2f} ({'above' if ratio > 1 else 'not above'} 1)"
    rows.append(f"| {key} | {show(f)} | {show(t)} | {gate} |")
  for key in ("lines of code", "clock"): rows.append(f"| {key} | {show(baretile['versions'].get(key) if key != 'clock' else baretile.get(key))} | "
                                                    f"{show(ttnn['versions'].get(key) if key != 'clock' else ttnn.get(key))} | |")
  return "\n".join(rows)

def main():
  if sys.argv[1] == "table": return print(table(*(json.load(open(p)) for p in sys.argv[2:4])))
  if sys.argv[1] == "load":
    path = os.environ.get("BOOK") or "tale-of-two-cities.txt.bz2"
    book = bz2.open(path, "rt", encoding="utf-8").read()
    return print(json.dumps(load(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32, SIZES[False].turns, book), indent=1))
  stack, folder, refpt, s = sys.argv[1], sys.argv[2], sys.argv[3], SIZES["--quick" in sys.argv]
  assert stack in ("baretile", "ttnn"), "the stack must be baretile or ttnn"
  prompt, after, top5 = llama.reference(refpt)
  path = os.environ.get("BOOK") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(refpt))), "tale-of-two-cities.txt.bz2")
  out = {"baretile": baretile, "ttnn": ttnn}[stack](folder, prompt, after, top5, bz2.open(path, "rt", encoding="utf-8").read(), s)
  json.dump(out, open(f"benchmark-{stack}.json", "w"), indent=1)
  for k, v in out.items(): print(f"{k}: {v}")

if __name__ == "__main__": main()
