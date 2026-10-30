# Bring baretile up on a card from the bottom of the stack. Each stage runs once the ones before it have passed, and
# the first failure stops the run, with what to look at next. Every stage's output goes in the record folder, and
# record.json describes the run: the device, the host, the commit, and each stage's result and time. docs/bring-up.md
# explains the stages.
#   python3 test/bringup.py record-folder [--smollm path/to/SmolLM-135M] [--llama path/to/Llama-3.1-8B-Instruct
#     --reference path/to/Llama-3.1-8B-Instruct.refpt] [--from stage] [--only stage,stage]
# On a card, run it from the checkout with TTSIM unset, and as root when tt-kmd doesn't drive the card. On ttsim every
# stage runs too, but the later ones take hours there. CI's card job runs it on a P150a (docs/bring-up.md). The SmolLM
# and Llama stages run when their files are given; the Llama one is the first workload's token-accuracy gate
# (docs/workload.md).
import argparse, json, os, pathlib, platform, re, shutil, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile import pci

ROOT = pathlib.Path(__file__).parent.parent
PY, TESTS = sys.executable, [sys.executable, "-m", "pytest", "-s", "-q"]
FIRST = [f"test/test_device.py::test_{t}" for t in ("config_space", "arc", "dma", "l1", "dram")]
MINUTE = 60

def accurate(log):  # the first workload's gate: at least 450 of 500 steps top-1, and 490 top-5
  m = re.search(r"top-1 ([\d.]+)%, top-5 ([\d.]+)%", log)
  return bool(m) and float(m[1]) >= 90 and float(m[2]) >= 98

# (name, what it checks, its commands, the minutes each may take, what to look at when it fails, and a test of its
# output beyond the commands' exit codes). A command is a function of the arguments, which gives None to skip the stage.
STAGES = [
  ("preflight", "clang, lld, and the card's sysfs without touching the card", [lambda a: [PY, "-m", "baretile.chip", "check"]], 1,
   "what check reports, and Before you start in docs/bring-up.md"),
  ("chip", "ARC, the AI clock, the Tensix columns and DRAM training", [lambda a: [PY, "-m", "baretile.chip"]], 5,
   "the card's board firmware, with Tenstorrent's tools"),
  ("transfers", "config space, ARC messages, and DMA, L1 and DRAM transfers that the host drives", [lambda a: TESTS + FIRST], 15,
   "the TLB windows and the DMA path. Nothing has run on the workers yet"),
  ("launches", "kernels on the workers, the dispatcher, and workers reaching host memory",
   [lambda a: [PY, "-m", "baretile.hello"], lambda a: TESTS + ["test/test_device.py"]], 30,
   "dispatch and multicast first, then the workers' reports and the status write to host memory (docs/testing.md)"),
  ("math", "instruction encoding and basic unpack, math and pack", [lambda a: TESTS + ["test/test_kernels.py", "-k", "tile_math or tensor_math"]], 30,
   "the Tensix registers and encodings, in What still needs checking in docs/bring-up.md"),
  ("probe", "launch overhead, transfers, memory latency and Tensix rates",
   [lambda a: [PY, "-m", "baretile.probe", "8", str(a.record / "probe.json")]], 60,
   "the probe that failed. Its numbers set tensor.DMA_UPLOAD and the cost model"),
  ("kernels", "every kernel and tensor operation against NumPy", [lambda a: TESTS + ["test/test_kernels.py"]], 120,
   "whether the failure comes and goes, which points at a wait that ttsim doesn't model (docs/testing.md, ttsim-chaos)"),
  ("models", "the models, the server and the adapters", [lambda a: TESTS + ["test/test_models.py"]], 180, "the model test that failed"),
  ("smollm", "SmolLM-135M's checkpoint against NumPy", [lambda a: a.smollm and TESTS + ["test/test_models.py::test_llama_model"]], 30,
   "the NumPy comparison that the test prints"),
  ("llama", "the first workload's token accuracy: top-1 90% and top-5 98% over 500 steps",
   [lambda a: a.llama and [PY, "-m", "baretile.models.llama", a.llama, a.reference, "500", "tt"]], 240,
   "the bf16 and NumPy runs that docs/workload.md describes, to tell weight precision from device arithmetic", accurate),
]

def run(args):
  args.record.mkdir(parents=True, exist_ok=True)
  def version(*cmd):  # the first line a tool prints about itself. it may be missing, which preflight then reports
    if not shutil.which(cmd[0]): return "missing"
    return (subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip().splitlines() or [""])[0]
  sim = pci.simulator()
  try: device = f"ttsim at {sim}" if sim else f"card {pci.find_cards()[0]}"
  except RuntimeError as e: device = str(e)
  record = dict(started=time.strftime("%Y-%m-%d %H:%M:%S %z"), device=device, host=platform.platform(), python=platform.python_version(),
                clang=version("clang", "--version"), commit=version("git", "describe", "--always", "--dirty"), stages=[])
  env = os.environ | ({"LLAMA": args.smollm} if args.smollm else {})  # test_llama_model reads SmolLM's folder from LLAMA
  names = [s[0] for s in STAGES]
  todo = names[names.index(args.start):] if args.start else names
  todo = [n for n in todo if n in args.only] if args.only else todo
  passed = True
  for name, checks, commands, minutes, hint, *test in STAGES:
    entry = dict(stage=name, checks=checks)
    record["stages"].append(entry)
    argv = [c(args) for c in commands]
    if name not in todo or not passed or None in argv or "" in argv:
      entry["result"] = "not run"
      continue
    log, start = args.record / f"{name}.log", time.monotonic()
    entry |= dict(commands=[" ".join(map(str, a)) for a in argv], log=log.name)
    with open(log, "w") as f:
      try:
        ok = all(subprocess.run(a, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env, timeout=minutes * MINUTE).returncode == 0 for a in argv)
        entry["result"] = "passed" if ok and all(t(log.read_text()) for t in test) else "failed"
      except subprocess.TimeoutExpired: entry["result"] = f"timed out after {minutes} minutes"
    entry["seconds"] = round(time.monotonic() - start)
    passed = entry["result"] == "passed"
    print(f"{name:10s}{entry['result']} in {entry['seconds']}s: {checks}", flush=True)
    if not passed: print(f"          {log} has its output. Look at {hint}", file=sys.stderr)
    (args.record / "record.json").write_text(json.dumps(record, indent=2) + "\n")
  (args.record / "record.json").write_text(json.dumps(record, indent=2) + "\n")
  return passed

if __name__ == "__main__":
  p = argparse.ArgumentParser(description="Bring baretile up on a card, from the bottom of the stack, and record each stage.")
  p.add_argument("record", type=pathlib.Path, help="the folder for record.json and each stage's log")
  p.add_argument("--smollm", help="a local HuggingFaceTB/SmolLM-135M folder")
  p.add_argument("--llama", help="a local Llama-3.1-8B-Instruct folder, for the first workload's accuracy")
  p.add_argument("--reference", help="Tenstorrent's Llama-3.1-8B-Instruct.refpt")
  p.add_argument("--from", dest="start", choices=[s[0] for s in STAGES], help="start at this stage")
  p.add_argument("--only", type=lambda s: s.split(","), help="run only these stages, separated by commas")
  args = p.parse_args()
  if bool(args.llama) != bool(args.reference): p.error("--llama and --reference go together")
  sys.exit(not run(args))
