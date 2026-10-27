# Tests for layer 4, what runs on baretile: the models, the server, training, the frontends and the benchmarks.
# conftest.py says how they run.
import json, math, os, pathlib, struct, sys, threading, time, unicodedata, pytest
import numpy as np
from types import SimpleNamespace
from baretile import tensor
from baretile.tensor import Tensor
from baretile.runtime import Runtime
from conftest import bf16, value
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "models"))  # for gpt2, llama, qwen, serve and cost
import benchmark, cost, gpt2, llama, qwen, serve, train

def test_train(chip):
  # A 2-block GPT learns to count in fp32 with Adam, and its steps match PyTorch's on the same weights. It doesn't
  # learn README and NOTES like train.py does, since those docs keep changing. Gradients may be off by 2e-3 of the
  # largest, which is the matmul's 2^-11 compounded. The step's kernels don't fit in L1 together, so a replay sends
  # them in parts.
  pytest.importorskip("torch")
  losses = train.train(Runtime(chip), 5, data=" ".join(map(str, range(3000))))
  assert all(abs(f - p) < 1e-3 and g < 2e-3 for f, p, g, _ in losses), losses
  assert losses[-1][0] < losses[0][0] - 0.25 and abs(losses[0][0] - losses[0][3]) < 1e-3, losses

def test_tinygrad_jit_unrecorded(chip):
  # TinyJit records a function's second call and replays it after that. A second call that can't be recorded, here
  # because an input isn't on the chip, runs as it is, and the next call that can be recorded is.
  if "TINYGRAD" in os.environ: sys.path.insert(0, os.environ["TINYGRAD"])
  pytest.importorskip("tinygrad")
  from adapters import tinygrad_baretile
  from tinygrad import Tensor, TinyJit
  rt = tinygrad_baretile.RT = tinygrad_baretile.RT or Runtime(chip)
  add, x = TinyJit(lambda x, y: (x + y.to("BARETILE")).realize()), Tensor([1.0, 2.0], device="BARETILE")
  for device in ("BARETILE", "CPU", "BARETILE", "BARETILE", "BARETILE"):
    assert add(x, Tensor([3.0, 4.0], device=device)).tolist() == [4.0, 6.0], device
  launches = rt.launches
  assert add(x, Tensor([5.0, 6.0], device="BARETILE")).tolist() == [6.0, 8.0] and rt.launches > launches

def test_tinygrad(chip):
  # tinygrad builds the graph and baretile runs it, checked against tinygrad's CPU device. Each case must launch on the
  # chip, so we know the host didn't compute it.
  if "TINYGRAD" in os.environ: sys.path.insert(0, os.environ["TINYGRAD"])  # TINYGRAD=/path/to/tinygrad, or installed
  pytest.importorskip("tinygrad")
  from adapters import tinygrad_baretile
  from tinygrad import Tensor, TinyJit, dtypes, function, nn
  from tinygrad.helpers import Context
  rt = tinygrad_baretile.RT = tinygrad_baretile.RT or Runtime(chip)  # the BARETILE device uses this process's chip
  rng = np.random.default_rng(14)
  a, b, w = (rng.standard_normal(shape).astype(np.float32) for shape in ((2, 3, 40), (2, 3, 40), (40, 5)))
  cases = [lambda x, y, w: x * 2 + y, lambda x, y, w: x.sum(1), lambda x, y, w: x.max(-1), lambda x, y, w: x.exp().log(),
           lambda x, y, w: (x > 0).where(x, y), lambda x, y, w: x @ w, lambda x, y, w: x.softmax(-1), lambda x, y, w: x.permute(2, 0, 1) - y.mean(),
           lambda x, y, w: x.pad(((0, 0), (1, 2), (0, 0))) * 1.5, lambda x, y, w: x[:, 1:3, ::2].relu(),
           lambda x, y, w: function(lambda p, q: p * 2 + q)(x, y)]
  for n, f in enumerate(cases):
    want = f(*(Tensor(v, device="CPU") for v in (a, b, w))).numpy()
    launches, got = rt.launches, f(*(Tensor(v, device="BARETILE") for v in (a, b, w))).numpy()
    assert rt.launches > launches and got.shape == want.shape, n
    assert np.abs(got - want).max() <= 1e-3 * (np.abs(want).max() + 1), (n, np.abs(got - want).max())  # matmuls are good to the FPU's 2^-11
  # free an 8-byte buffer. The next one, this int64, must not see what it held
  Tensor.ones(2, device="BARETILE").contiguous().realize().tolist()
  assert Tensor.ones((), dtype=dtypes.int64, device="BARETILE").contiguous().item() == 1
  ints = Tensor([[1, -8, 1], [32, 1, 6]], dtype=dtypes.int32, device="BARETILE")  # products come back int32. bitwise and runs on the host
  assert (ints * ints).tolist() == [[1, 64, 1], [1024, 1, 36]] and (ints & 12).tolist() == [[0, 8, 0], [0, 0, 4]]
  assert ints.argmax(1).tolist() == [0, 0]
  made = Tensor([3, 3], dtype=dtypes.int64, device="BARETILE").cast(dtypes.bfloat16)  # the host makes this bf16 and uploads its values
  assert (made * Tensor([1.0, 2.0], dtype=dtypes.bfloat16, device="BARETILE")).tolist() == [3.0, 6.0]
  assert (Tensor(a, device="BARETILE") > 0).any(1).tolist() == (a > 0).any(1).tolist()  # bools read back a byte each
  assert (Tensor(np.arange(8, dtype=np.float32), device="BARETILE")[:4] * 2).tolist() == [0, 2, 4, 6]  # a view of part of a tensor
  assert (ints.T @ ints).tolist() == (np.array(ints.tolist()).T @ np.array(ints.tolist())).tolist()  # summed in fp32, no FPU
  t = Tensor(a, device="BARETILE").realize()
  assert t[1, 1:3].numpy().tolist() == a[1, 1:3].tolist()  # reads a buffer through a tinygrad view of part of it (_offset)
  assert t[:, :0].sum(1).numpy().tolist() == [[0] * 40] * 2 and t[:, :0].numpy().shape == (2, 0, 40)  # nothing to compute
  assert (t[0, 0, :1].pad(((1, 0),))[0:1] * t[0, 0]).numpy().tolist() == [0] * 40  # padding stays padding, broadcast too
  t, want = Tensor.zeros(6, 6, device="BARETILE").contiguous().realize(), np.zeros((6, 6))  # stores into parts of a buffer
  t[2:4, 3:5] = Tensor.ones(2, 2, device="BARETILE"); t[0] = 7.0; t[1:6:2, 0] = Tensor([1.0, 2.0, 3.0], device="BARETILE")
  want[2:4, 3:5], want[0], want[1:6:2, 0] = 1, 7, [1, 2, 3]
  assert t.numpy().tolist() == want.tolist()
  def train(device):
    rng = np.random.default_rng(15)
    l1, l2 = nn.Linear(16, 32), nn.Linear(32, 10)
    for l in (l1, l2):
      l.weight = Tensor(rng.standard_normal(l.weight.shape).astype(np.float32) * 0.2, device=device)
      l.bias = Tensor.zeros(*l.bias.shape, device=device)
    opt, losses = nn.optim.Adam([l1.weight, l1.bias, l2.weight, l2.bias], lr=1e-2), []
    x, y = (Tensor(v, device=device) for v in (rng.standard_normal((8, 16)).astype(np.float32), rng.integers(0, 10, 8).astype(np.int32)))
    with Context(TRAINING=1):
      for _ in range(3):
        opt.zero_grad()
        losses.append(l2(l1(x).relu()).sparse_categorical_crossentropy(y).backward().item())
        opt.step()
    return losses
  want, got = train("CPU"), train("BARETILE")
  assert np.allclose(got, want, atol=1e-3) and got[-1] < got[0] - 0.3, (got, want)
  def jitted(device):  # the same under TinyJit, which replays from the third call on without running step
    rng, calls = np.random.default_rng(15), []
    l1 = nn.Linear(16, 10)
    l1.weight, l1.bias = Tensor(rng.standard_normal((10, 16)).astype(np.float32) * 0.2, device=device), Tensor.zeros(10, device=device).contiguous()
    opt = nn.optim.Adam([l1.weight, l1.bias], lr=1e-2)
    @TinyJit
    def step(x, y):
      calls.append(1)
      with Context(TRAINING=1):
        opt.zero_grad()
        loss = l1(x).sparse_categorical_crossentropy(y).backward()
        opt.step()
        return loss.realize()
    data = [(rng.standard_normal((8, 16)).astype(np.float32), rng.integers(0, 10, 8).astype(np.int32)) for _ in range(5)]
    return [step(Tensor(x, device=device).realize(), Tensor(y, device=device).realize()).item() for x, y in data], len(calls)
  (want, _), (got, calls) = jitted("CPU"), jitted("BARETILE")
  assert np.allclose(got, want, atol=1e-3) and calls == 2, (got, want, calls)

def test_torch_frontend(chip):
  # PyTorch runs on BARETILE through tinygrad's torch frontend (extra/torch_backend, PyTorch's "tiny" device), checked
  # against PyTorch on the CPU. The matmul and conv are good to the FPU's 2^-11.
  if "TINYGRAD" in os.environ: sys.path.insert(0, os.environ["TINYGRAD"])
  pytest.importorskip("tinygrad")
  pytest.importorskip("extra.torch_backend", reason="needs a tinygrad checkout in TINYGRAD, since an installed tinygrad has no extra/")
  torch = pytest.importorskip("torch")
  from adapters import tinygrad_baretile
  from tinygrad.helpers import Context
  rt = tinygrad_baretile.RT = tinygrad_baretile.RT or Runtime(chip)
  with Context(DEV="BARETILE"):  # the tinygrad device behind "tiny", read at each op
    import extra.torch_backend.backend  # noqa: F401 -- registers "tiny" with PyTorch. its C++ part is built once and cached
    g = torch.Generator().manual_seed(0)
    a, b, w = torch.randn(4, 40, generator=g), torch.randn(4, 40, generator=g), torch.randn(40, 24, generator=g)
    image, kernel, norm = torch.randn(1, 2, 10, 10, generator=g), torch.randn(3, 2, 3, 3, generator=g), torch.nn.LayerNorm(40)
    cases = [lambda x, y, w: (x * 2 + y).relu() - y.exp() / 4, lambda x, y, w: x.sum(1), lambda x, y, w: x @ w,
             lambda x, y, w: x.softmax(-1), lambda x, y, w: norm.to(x.device)(x),
             lambda x, y, w: torch.nn.functional.conv2d(image.to(x.device), kernel.to(x.device))]
    for n, f in enumerate(cases):
      want = f(a, b, w).detach()
      launches, got = rt.launches, f(*(t.to("tiny") for t in (a, b, w))).detach().cpu()
      assert rt.launches > launches and got.shape == want.shape, n
      assert (got - want).abs().max() <= 1e-3 * (want.abs().max() + 1), (n, (got - want).abs().max())
    def train(device):
      torch.manual_seed(1)
      model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10)).to(device)
      opt, g = torch.optim.Adam(model.parameters(), lr=1e-2), torch.Generator().manual_seed(2)
      x, y = torch.randn(8, 16, generator=g).to(device), torch.randint(0, 10, (8,), generator=g).to(device)
      losses = []
      for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
      return losses
    want, got = train("cpu"), train("tiny")
    assert np.allclose(got, want, atol=1e-4) and got[-1] < got[0] - 0.2, (got, want)

def test_tinygrad_llm(chip):
  # The Transformer in tinygrad/llm runs two SmolLM-shaped blocks on BARETILE, against our numpy Llama with the same
  # bf16 weights. tinygrad's KV cache is fp16, which the adapter doesn't have, so it stays on the host. The blocks are
  # precompiled tinygrad functions, and the adapter inlines them with fresh buffers on each call.
  if "TINYGRAD" in os.environ: sys.path.insert(0, os.environ["TINYGRAD"])
  pytest.importorskip("tinygrad.llm.model")
  from adapters import tinygrad_baretile
  from tinygrad import Tensor as TT, dtypes, nn
  from tinygrad.helpers import Context
  from tinygrad.llm.model import Transformer, TransformerConfig
  rt = tinygrad_baretile.RT = tinygrad_baretile.RT or Runtime(chip)
  c, w = smol(np.random.default_rng(37), 2)
  sd = {"token_embd.weight": w["embed"], "output.weight": w["embed"], "output_norm.weight": w["norm"][0]}
  for l in range(2):  # tinygrad's Linear is (out, in) like Hugging Face's, and keeps q, k, v, gate and up apart
    qkv, gate_up, p = w[f"layers.{l}.qkv_proj"].T, w[f"layers.{l}.gate_up_proj"].T, f"layers.{l}."
    sd |= {f"blk.{l}.{k}": v for k, v in (("attn_q.weight", qkv[:576]), ("attn_k.weight", qkv[576:768]), ("attn_v.weight", qkv[768:]),
           ("attn_output.weight", w[p + "o_proj"].T), ("ffn_gate.weight", gate_up[:1536]), ("ffn_up.weight", gate_up[1536:]),
           ("ffn_down.weight", w[p + "down_proj"].T), ("attn_norm.weight", w[p + "input_layernorm"][0]),
           ("ffn_norm.weight", w[p + "post_attention_layernorm"][0]))}
  with Context(DEV="BARETILE"):
    model = Transformer(TransformerConfig(num_blocks=2, dim=576, hidden_dim=1536, n_heads=9, n_kv_heads=3, norm_eps=c.eps, vocab_size=256,
                                          head_dim=64, rope_theta=c.theta, rope_dim=64, v_head_dim=64, max_context=64))
    nn.state.load_state_dict(model, {k: TT(np.ascontiguousarray(v, np.float32)).cast(dtypes.bfloat16) for k, v in sd.items()}, verbose=False)
    def logits(ids, start):  # Transformer.forward without its sampling
      x = model.token_embd(TT([ids], dtype=dtypes.int32)).float()
      for b in model.blk: x = b(x, start)
      return model.output(model.output_norm(x[:, -1:]))[0, -1].numpy()
    np_kv = llama.kv_cache(c, 64, np.zeros)
    for ids, places in (([7, 200, 31, 64, 5], range(5)), ([42], [5])):
      launches, got, want = rt.launches, logits(ids, places[0]), llama.forward(ids, places, w, c, np_kv)[0]
      assert rt.launches - launches > 50 and np.abs(got - want).max() < 0.02 * np.ptp(want), (np.abs(got - want).max(), np.ptp(want))

@pytest.mark.parametrize("dtype", ["bfp8", "bfp4"])
def test_bfp_numpy(chip, dtype):
  # gpt2.bfp, numpy's model of a block float, must match the chip's cast bit for bit, with exponents spread wide in
  # each run. The numpy models rely on it to round values the way the chip holds them.
  rt, rng = Runtime(chip), np.random.default_rng(5)
  a = bf16(rng.normal(0, 1, (64, 96)) * np.exp(rng.normal(0, 2, (64, 96))))
  assert np.array_equal(value(Tensor.of(rt, a.tolist()).cast(dtype))[0], gpt2.bfp(a, dtype))

@pytest.mark.parametrize("n, dtype, size", [(16, "bf16", 0), (40, "bf16", 0), (40, "bfp8", 0), (16, "bf16", 64)])
def test_gpt2_layer(chip, n, dtype, size):
  # A GPT-2 small block on n tokens, against the same code in numpy float64. 40 tokens are two tile rows, so attention
  # spans tiles. With size, keys and values go into a KV cache of that many positions, the empty ones masked. Biases
  # and norms get 0.1 of spread so they matter. The intermediates are bf16 here, so the test allows 3% error in the
  # block's change to x, which is up to about 2.
  rt, rng, d = Runtime(chip), np.random.default_rng(3), 768
  shapes = {"ln_1.weight": (1, d, 1), "ln_1.bias": (1, d, 0), "attn.c_attn.weight": (d, 3 * d, 0), "attn.c_attn.bias": (1, 3 * d, 0),
            "attn.c_proj.weight": (d, d, 0), "attn.c_proj.bias": (1, d, 0), "ln_2.weight": (1, d, 1), "ln_2.bias": (1, d, 0),
            "mlp.c_fc.weight": (d, 4 * d, 0), "mlp.c_fc.bias": (1, 4 * d, 0), "mlp.c_proj.weight": (4 * d, d, 0), "mlp.c_proj.bias": (1, d, 0)}
  p = {k: bf16(rng.normal(mean, 0.1 if r == 1 else 0.02, (r, c))) for k, (r, c, mean) in shapes.items()}
  x = bf16(rng.normal(0, 1, (n, d)))
  t = {k: Tensor.of(rt, v.tolist()) for k, v in p.items()}
  t |= {k: v.cast(dtype).realize() for k, v in t.items() if v.rows > 1}
  p |= {k: gpt2.values(v) for k, v in t.items() if v.dtype != "bf16"}
  cache = (lambda zeros: zeros((24, size, 64)) if size else None)  # a block's keys and values, or none
  places = Tensor.of(rt, [list(range(n))], "int32") if size else None
  got = gpt2.layer(Tensor.of(rt, x.tolist()), t, Tensor.of(rt, gpt2.mask(range(n), size or n).tolist()), 12,
                   cache(lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), places)
  got = np.array(got.tolist())
  want = gpt2.layer(x, p, gpt2.mask(range(n), size or n), 12, cache(np.zeros), range(n))
  change = np.abs(want - x).max()
  assert change > 1 and np.abs(got - want).max() < 0.03 * change, (np.abs(got - want).max(), change)

@pytest.mark.skipif("GPT2" not in os.environ, reason="GPT2=/path/to/model.safetensors (Hugging Face openai-community/gpt2) runs it")
def test_gpt2_model(chip):
  # GPT-2 small with its pretrained weights, on a prompt and 3 decoded tokens, against numpy with the same bf16
  # weights. In fp32, numpy matches Hugging Face to 1e-4. The chip's pick has to be a token that numpy rates as good,
  # not necessarily numpy's. numpy picks " not" (407) with " to" 0.36 behind, which bf16 can't tell apart through 12
  # blocks. The residual reaches 3000, where bf16 steps by 16. It takes about 90 s on ttsim.
  rt = Runtime(chip)
  weights = gpt2.load(os.environ["GPT2"])
  t, w, ids = gpt2.upload(rt, weights), gpt2.arrays(weights), [464, 3616, 286, 1204, 318]  # tiktoken's gpt2 encoding
  kv = gpt2.kv_cache(12, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist()))
  run, on = lambda ids, places: gpt2.forward(ids, t, gpt2.mask(places, 32), places, kv), lambda row: Tensor.of(rt, [row], "int32")
  step = tensor.jit(run)
  for i in range(4):
    new = ids if i == 0 else ids[-1:]
    got = gpt2.values((run if i == 0 else step)(on(new), on(list(range(len(ids) - len(new), len(ids))))))[0]
    want = gpt2.logits(ids, w)
    err = np.abs(got - want).max()
    assert (i or want.argmax() == 407) and err < 2 and want[got.argmax()] >= want.max() - 2 * err, (i, err, got.argmax())
    ids.append(int(want.argmax()))

def smol(rng, layers, vocab=256, places=64):  # SmolLM-135M's config, and random weights in its shapes with llama.py's names
  c = llama.model(dim=576, heads=9, kv=3, layers=layers, eps=1e-5, theta=10000.0)
  shapes = {"input_layernorm": (1, 576, 1), "post_attention_layernorm": (1, 576, 1), "q_proj": (576, 576, 0), "k_proj": (576, 192, 0),
            "v_proj": (576, 192, 0), "o_proj": (576, 576, 0), "gate_proj": (576, 1536, 0), "up_proj": (576, 1536, 0), "down_proj": (1536, 576, 0)}
  w = {f"layers.{l}.{k}": bf16(rng.normal(mean, 0.1 if r == 1 else 0.02, (r, n))) for l in range(layers) for k, (r, n, mean) in shapes.items()}
  w |= {"norm": bf16(rng.normal(1, 0.1, (1, 576))), "embed": bf16(rng.normal(0, 1, (vocab, 576)))}
  return c, llama.fuse(w | {"embed.T": w["embed"].T.copy()} | dict(zip(("cos", "sin", "turn"), llama.rotary(c, places))), c)

def near(got, want):  # each row picks numpy's best, or a token within 1% of the span, which bf16 activations can't tell apart
  return all(w.max() - w[g.argmax()] <= 0.01 * np.ptp(w) for g, w in zip(got, want))

def qwen35(rng, vocab=256, places=64):
  """A small random Qwen3.5-MoE and its config, with bf16 Weights named and fused the way llama.load gives them.

  It has 3 Gated DeltaNet blocks, a gated attention block, and the multi-token prediction layer as block 4. Tokens use
  all 8 experts, so a near tie between experts can't decide a test. The embedding is small next to what the
  blocks add, so losing the Gated DeltaNet states moves the logits by 100%."""
  c = llama.model(dim=128, heads=4, kv=2, head_dim=64, layers=4, eps=1e-6, theta=1e7, partial=0.25, gated=True, centered=True, conv=4,
                  kh=2, vh=4, kd=32, vd=32, k=8, norm_topk=True, layer_types=["linear_attention"] * 3 + ["full_attention"], mtp=1)
  bits = lambda *shape, mean=0.0, sd=0.05: llama.Weight(lambda a=gpt2.bf16_bits(rng.normal(mean, sd, shape)): a)
  # embed.T is a separate output layer, untied from embed
  w = {"embed": bits(vocab, 128, sd=0.3), "embed.T": bits(128, vocab), "norm": bits(1, 128, sd=0.3), "mtp.fc": bits(256, 128),
       "mtp.pre_fc_norm_embedding": bits(1, 128, sd=0.3), "mtp.pre_fc_norm_hidden": bits(1, 128, sd=0.3), "mtp.norm": bits(1, 128, sd=0.3)}
  for l in range(c.layers + c.mtp):  # block 4, the multi-token prediction layer, is an attention block
    pre = f"layers.{l}."
    w |= {pre + "input_layernorm": bits(1, 128, sd=0.3), pre + "post_attention_layernorm": bits(1, 128, sd=0.3), pre + "gate": bits(8, 128),
          pre + "experts.gate_up_proj": bits(8, 128, 128), pre + "experts.down_proj": bits(8, 128, 64),
          pre + "shared_expert.gate_proj": bits(128, 64), pre + "shared_expert.up_proj": bits(128, 64),
          pre + "shared_expert.down_proj": bits(64, 128), pre + "shared_expert_gate": bits(1, 128)}
    if l < 3: w |= {pre + "in_proj_qkv": bits(128, 256), pre + "in_proj_z": bits(128, 128), pre + "in_proj_b": bits(128, 4),
                    pre + "in_proj_a": bits(128, 4), pre + "conv1d": bits(256, 1, 4, sd=0.3), pre + "A_log": bits(1, 4, sd=0.5),
                    pre + "dt_bias": bits(1, 4, mean=1.0, sd=0.2), pre + "norm": bits(1, 32, mean=1.0, sd=0.2), pre + "out_proj": bits(128, 128)}
    else: w |= {pre + "q_proj": bits(128, 512), pre + "k_proj": bits(128, 128), pre + "v_proj": bits(128, 128), pre + "o_proj": bits(256, 128),
                pre + "q_norm": bits(1, 64, sd=0.3), pre + "k_norm": bits(1, 64, sd=0.3)}
  w |= {k: llama.Weight(lambda a=a: gpt2.bf16_bits(a)) for k, a in zip(("cos", "sin", "turn"), llama.rotary(c, places))}
  return c, llama.fuse(w, c)

def test_llama_frequencies():
  # Llama 3.1's RoPE scaling slows long wavelengths by 8, keeps short ones, and blends those between. Our frequencies
  # must match Hugging Face's float32 ones bit for bit.
  m, transformers = pytest.importorskip("transformers.modeling_rope_utils"), pytest.importorskip("transformers")
  rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  hf = transformers.LlamaConfig(hidden_size=4096, num_attention_heads=32, rope_theta=500000.0, rope_scaling=rope, max_position_embeddings=131072)
  want = m.ROPE_INIT_FUNCTIONS["llama3"](hf, "cpu")[0].numpy()
  got = llama.frequencies(llama.model(dim=4096, heads=32, theta=500000.0, rope=rope))
  assert got.dtype == np.float32 and np.array_equal(got, want), np.abs(got - want).max()
  plain = llama.frequencies(llama.model(dim=4096, heads=32, theta=500000.0))
  assert plain[0] == got[0] and plain[-1] == 8 * got[-1], "short wavelengths kept, long ones slowed by the factor"

def safetensors(path, tensors):  # writes {name: (dtype "BF16" or "F32", array of bits or values)} as a safetensors file
  header, data = {}, b""
  for name, (dtype, a) in tensors.items():
    header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [len(data), len(data) + a.nbytes]}
    data += a.tobytes()
  head = json.dumps(header).encode()
  pathlib.Path(path).write_bytes(struct.pack("<Q", len(head)) + head + data)

def test_llama_load(tmp_path):
  # A checkpoint split like Llama-3.1-8B's, in two files and an index, with its own lm_head and a float32 norm. The
  # RoPE tables cover the 64 positions asked for, not the model's 131072.
  rng, dim, ffn, vocab = np.random.default_rng(31), 64, 96, 40
  w = lambda r, c: (rng.standard_normal((r, c)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
  parts = {"model.embed_tokens.weight": w(vocab, dim), "lm_head.weight": w(vocab, dim), "model.norm.weight": w(1, dim)[0],
           **{f"model.layers.0.self_attn.{n}_proj.weight": w(r, dim) for n, r in (("q", dim), ("k", 32), ("v", 32), ("o", dim))},
           **{f"model.layers.0.mlp.{n}_proj.weight": w(r, cl) for n, r, cl in (("gate", ffn, dim), ("up", ffn, dim), ("down", dim, ffn))}}
  norms = {f"model.layers.0.{n}.weight": rng.standard_normal(dim).astype(np.float32) for n in ("input_layernorm", "post_attention_layernorm")}
  files = [{k: ("BF16", v) for k, v in list(parts.items())[:5]} | {k: ("F32", v) for k, v in norms.items()},
           {k: ("BF16", v) for k, v in list(parts.items())[5:]}]
  for i, file in enumerate(files): safetensors(tmp_path / f"model-0000{i + 1}-of-00002.safetensors", file)
  weight_map = {k: f"model-0000{i + 1}-of-00002.safetensors" for i, file in enumerate(files) for k in file}
  (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
  rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  (tmp_path / "config.json").write_text(json.dumps({"hidden_size": dim, "num_attention_heads": 2, "num_key_value_heads": 1, "num_hidden_layers": 1,
                                                    "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "max_position_embeddings": 131072,
                                                    "rope_scaling": rope, "vocab_size": vocab, "intermediate_size": ffn}))
  c, weights = llama.load(str(tmp_path), 64)
  assert all(isinstance(x, llama.Weight) for x in weights.values())
  assert c.places == 64 and c.rope == rope and c.ffn == ffn and weights["cos"].bits().shape == (64, 4 * 32)
  q, k, v, gate, up = (parts[f"model.layers.0.{n}_proj.weight"] for n in ("self_attn.q", "self_attn.k", "self_attn.v", "mlp.gate", "mlp.up"))
  assert np.array_equal(weights["layers.0.qkv_proj"].bits(), np.concatenate([q.T, k.T, v.T], 1))
  assert np.array_equal(weights["layers.0.gate_up_proj"].bits(), np.concatenate([gate.T, up.T], 1))
  assert np.array_equal(weights["embed.T"].bits(), parts["lm_head.weight"].T), "its own output layer, not the embedding"
  assert np.array_equal(weights["layers.0.input_layernorm"].bits()[0], gpt2.bf16_bits(norms["model.layers.0.input_layernorm.weight"]))
  out = llama.forward([3, 7], [0, 1], llama.arrays(weights), c, llama.kv_cache(c, 32, np.zeros))
  assert out.shape == (1, vocab) and np.isfinite(out).all()

def test_tokenizer(tmp_path):
  # llama.Tokenizer must encode like tiktoken over the same tokenizer.json, and decode back, with the pre-tokenizers of
  # Llama 3, SmolLM, Qwen (NFC first) and gpt-oss. It spells out \p{L}, \p{N} and \s for Python's re, whose \s also
  # matches four control characters. "\x1cthe" catches that. A piece that's in the vocabulary is taken as is, like the
  # lone emoji, whose bytes no merge joins.
  tiktoken = pytest.importorskip("tiktoken")
  bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))  # GPT-2's bytes as printable characters
  char = dict(zip(bs + [b for b in range(256) if b not in bs], map(chr, bs + [256 + i for i in range(256 - len(bs))])))
  tokens = [bytes([b]) for b in range(256)] + [b"th", b"he", b"the", b" the", b"in", b"ing", b"12", b"123", b"\r\n", b"\x1cthe",
                                               "é".encode(), "😀".encode()]
  vocab = {"".join(char[b] for b in t): i for i, t in enumerate(tokens)}  # ids in merge order
  text = ("The thing's THE THING'LL 1234567 x²³ ½ ٣٤٥ in\u00a0the\u2003end \t\tthing\r\n\r\nthe \x1c\x1dctrl \x1cthe café "
          "😀😀 x😀y e\u0301 McDonald's ǅungla ʰa/\n")
  llama3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  o200k = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?"
           r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
           r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
  for split, nfc in ((llama3, False), (None, False), (llama3, True), (o200k, False)):
    pre = {"type": "Split", "pattern": {"Regex": split}} if split else {"type": "ByteLevel"}
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": [{"content": "<|end|>", "id": len(vocab)}],
                                                         "pre_tokenizer": pre, "normalizer": {"type": "NFC"} if nfc else None}))
    ours = llama.Tokenizer(str(tmp_path))
    theirs = tiktoken.Encoding("t", pat_str=split or llama.SPLIT, mergeable_ranks=dict(zip(tokens, range(len(tokens)))),
                               special_tokens={"<|end|>": len(vocab)})
    plain = unicodedata.normalize("NFC", text) if nfc else text
    assert ours.encode(text) == theirs.encode(plain), (split, nfc)
    assert ours.decode(ours.encode(text) + [len(vocab)]) == plain + "<|end|>"

@pytest.mark.parametrize("n", [16, 40])
def test_llama_layer(chip, n):
  # A Llama block in SmolLM-135M's shapes on n tokens, against the same code in numpy float64. The intermediates are
  # bf16 here, so the test allows 3% error in the block's change to x.
  rt, rng = Runtime(chip), np.random.default_rng(12)
  c, w = smol(rng, 1)
  p = {k.removeprefix("layers.0."): v for k, v in w.items() if k.startswith("layers.0.")}
  x, (cos, sin, turn) = bf16(rng.normal(0, 1, (n, 576))), llama.rotary(c, n)
  mask = gpt2.mask(range(n), n)
  on = lambda a: Tensor.of(rt, a.tolist())
  cache, np_cache = on(np.zeros((6, n, 64))), np.zeros((6, n, 64))
  places = Tensor.of(rt, [list(range(n))], "int32")
  got = llama.layer(on(x), {k: on(v) for k, v in p.items()}, c, (on(cos), on(sin), on(turn)), (on(mask), places.reshape(1, n, 1), 0), cache, places)
  want = llama.layer(x, p, c, (np.float64(cos), np.float64(sin), turn), np.float64(mask), np_cache, range(n))
  change = np.abs(want - x).max()
  assert change > 0.5 and np.abs(np.array(got.tolist()) - want).max() < 0.03 * change, (np.abs(np.array(got.tolist()) - want).max(), change)

def test_kv_cache(chip):
  # A two-block Llama decodes 5 tokens after a 5-token prompt, against numpy. Positions are data, not shapes, so the
  # steps launch the same kernels and replay from the third.
  rt, rng = Runtime(chip), np.random.default_rng(13)
  c, w = smol(rng, 2)
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  kv, np_kv = llama.kv_cache(c, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 32, np.zeros)
  on = lambda row: Tensor.of(rt, [row], "int32")
  step, ids = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv)), [7, 200, 31, 64, 5]
  for i in range(6):
    new = ids if i == 0 else ids[-1:]
    places = list(range(len(ids) - len(new), len(ids)))
    got = llama.logits(llama.forward(on(new), on(places), t, c, kv) if i == 0 else step(on(new), on(places)))
    want = llama.forward(new, places, w, c, np_kv)[0]
    assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), (i, np.abs(got - want).max(), np.abs(want).max())
    ids.append(int(want.argmax()))
  assert np.allclose(want, llama.forward(ids[:-1], range(len(ids) - 1), w, c, llama.kv_cache(c, 32, np.zeros))[0], atol=1e-3)

def test_moe(chip):
  # A mixture of experts (qwen.moe) against numpy, with the experts picked and their weights read on the chip. Up to
  # 40 tokens use qwen.touched, and 160 use qwen.group, which must replay bit for bit. A free slot (live 0) gets output
  # 0. A two-block model must decode the same from a replay as op by op, so a replayed step reads the experts its
  # token chose.
  rt, rng = Runtime(chip), np.random.default_rng(31)
  experts = lambda dim, n, m: {"router": bf16(rng.normal(0, 0.5, (dim, n))), "experts.gate_up": bf16(rng.normal(0, 0.1, (n, dim, 2 * m))),
                               "experts.down": bf16(rng.normal(0, 0.1, (n, m, dim)))}
  c, w = llama.model(dim=64, k=4), experts(64, 16, 64)
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  for n in (1, 3, 8, 40, 160):  # row 1, if there is one, is a free slot
    h, live = bf16(rng.normal(0, 1, (n, 64))), np.float32(np.arange(n) != 1).reshape(n, 1)
    got, want = value(qwen.moe(Tensor.of(rt, h.tolist()), t, c, live=Tensor.of(rt, live.tolist())))[0], qwen.moe(h, w, c, live=live)
    assert np.abs(got - want).max() < 0.02 * np.abs(want).max() and not got[1:2].any(), (n, np.abs(got - want).max())
  ids, weights = Tensor.of(rt, [[0, 3], [5, 3], [7, 1]], "int32"), Tensor.of(rt, [[0.5, 0.5], [0, 0], [0.25, 0.75]])
  assert value(qwen.touched(ids, 8, weights)[0])[0, :, 0].tolist() == [0, 1, 3, 7, 8, 8], "expert 5, weighted 0, not listed"
  grouped, prompts = tensor.jit(lambda h: qwen.moe(h, t, c)), np.random.default_rng(41)  # each replayed call does its own grouping
  for _ in range(3):
    h = Tensor.of(rt, bf16(prompts.normal(0, 1, (160, 64))).tolist())
    assert np.array_equal(value(grouped(h)), value(qwen.moe(h, t, c))), "replayed expert grouping must match a fresh call"
  c, w = smol(rng, 2)
  c.k = 2
  for l in range(2):
    del w[f"layers.{l}.gate_up_proj"], w[f"layers.{l}.down_proj"]
    w |= {f"layers.{l}.{k}": v for k, v in experts(576, 8, 64).items()}
  t, on = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, lambda row: Tensor.of(rt, [row], "int32")
  def decode(replay):  # logits for a prompt and the 4 tokens numpy picks after it, and numpy's logits
    kv, np_kv = llama.kv_cache(c, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 32, np.zeros)
    step, ids, out = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv)), [1, 2, 3, 4, 5], []
    for i in range(5):
      new = ids if i == 0 else ids[-1:]
      places = list(range(len(ids) - len(new), len(ids)))
      out.append((llama.logits(llama.forward(on(new), on(places), t, c, kv) if i == 0 or not replay else step(on(new), on(places))),
                  want := llama.forward(new, places, w, c, np_kv)[0]))
      ids.append(int(want.argmax()))
    return out
  # In this prompt, each token's k-th and next router logits are 6% of their span apart, so bf16 rounding leaves the
  # chip picking numpy's experts. The experts' outputs are large and of either sign, so the sums lose more bits than in
  # a dense MLP, and the tolerance is 6%.
  replayed, alone = decode(True), decode(False)
  assert all(np.array_equal(a[0], b[0]) for a, b in zip(replayed, alone)), "replayed decode must match a fresh call"
  assert all(np.abs(got - want).max() < 0.06 * np.abs(want).max() for got, want in alone), [np.abs(g - w).max() / np.abs(w).max() for g, w in alone]
  held = []  # topk and group free the DRAM they take, so the amount held stays the same across calls
  for _ in range(3):
    qwen.group(Tensor.of(rt, rng.normal(0, 1, (40, 16)).tolist()).topk(4)[1], 16)
    rt.wait(), rt.reclaim()
    held.append(sum(rt.used.values()))
  assert held[0] == held[1] == held[2], held

def test_qwen3_moe(chip, tmp_path):
  # Qwen3-MoE from Hugging Face's files, against transformers in float32 with random bf16 weights. The experts are
  # weighted by their probabilities among all experts, since norm_topk_prob is off by default. numpy must be within
  # 0.5% (0.12% measured), and the chip within 2% (1%) with the same picks. At TT's performance precision the chip must
  # be within 1% of the span of numpy with the same block floats (0.8%).
  torch, transformers = pytest.importorskip("torch"), pytest.importorskip("transformers")
  torch.manual_seed(0)
  config = transformers.Qwen3MoeConfig(hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                       num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2, decoder_sparse_step=2,
                                       vocab_size=256, max_position_embeddings=64, rope_theta=1e6, tie_word_embeddings=False)
  model = transformers.Qwen3MoeForCausalLM(config).to(torch.bfloat16).float()
  model.save_pretrained(tmp_path)
  ids = [3, 1, 4, 1, 5, 9, 2, 6]
  with torch.no_grad(): want = model(torch.tensor([ids])).logits[0].numpy()
  (tmp_path / "config.json").write_text(json.dumps(json.loads((tmp_path / "config.json").read_text()) | {"num_hidden_layers": 1}))
  assert not any(k.startswith("layers.1.") for k in llama.load(str(tmp_path), 32)[1]), "a config cut to one block loads only that block"
  (tmp_path / "config.json").write_text(json.dumps(json.loads((tmp_path / "config.json").read_text()) | {"num_hidden_layers": 2}))
  c, weights = llama.load(str(tmp_path), 32)
  names = {k.split(".", 2)[2] for k in weights if k.startswith("layers.1.")}
  assert (c.k, llama.head(c)) == (2, 32) and {"router", "experts.gate_up", "experts.down", "qk_norm", "qk_on"} <= names
  rel = lambda got: max(np.abs(got[r] - want[r]).max() / np.abs(want[r]).max() for r in range(len(ids)))
  assert rel(llama.forward(ids, list(range(8)), llama.arrays(weights), c, llama.kv_cache(c, 32, np.zeros), every=True)) < 0.005
  rt = Runtime(chip)
  on = lambda row: Tensor.of(rt, [row], "int32")
  got = gpt2.values(llama.forward(on(ids), on(list(range(8))), llama.upload(rt, weights), c, llama.kv_cache(c, 32, llama.zeros(rt)), every=True))
  assert rel(got) < 0.02 and np.array_equal(got.argmax(-1), want.argmax(-1)), rel(got)
  c.precision = llama.precision(c, "tt")  # the experts' gate and up in bfp4 at one phase, their down in bfp8 at two
  t = llama.upload(rt, weights, c.precision)
  assert (t["layers.1.experts.gate_up"].dtype, t["layers.1.experts.down"].dtype, t["layers.1.router"].dtype) == ("bfp4", "bfp8", "bf16")
  got = gpt2.values(llama.forward(on(ids), on(list(range(8))), t, c, llama.kv_cache(c, 32, llama.zeros(rt, "bfp8")), every=True))
  want = llama.forward(ids, list(range(8)), llama.arrays(weights, c.precision), c, llama.kv_cache(c, 32, np.zeros), every=True)
  assert np.array_equal(got.argmax(-1), want.argmax(-1)) and np.abs(got - want).max() < 0.01 * np.ptp(want), np.abs(got - want).max() / np.ptp(want)

FAMILIES = {"qwen3": ("Qwen3", dict(head_dim=64)), "qwen2": ("Qwen2", {}), "mistral": ("Mistral", dict(sliding_window=None)),
            "qwen2_moe": ("Qwen2Moe", dict(moe_intermediate_size=64, shared_expert_intermediate_size=96, num_experts=8, num_experts_per_tok=2)),
            "mixtral": ("Mixtral", dict(num_local_experts=8, num_experts_per_tok=2)),
            "gpt_oss": ("GptOss", dict(head_dim=32, num_local_experts=8, num_experts_per_tok=2, sliding_window=4,
                                       layer_types=["sliding_attention", "full_attention"],
                                       rope_parameters={"rope_type": "yarn", "factor": 8.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False,
                                                        "original_max_position_embeddings": 8, "rope_theta": 1e4}))}

@pytest.mark.parametrize("family", FAMILIES)
def test_hf_families(chip, tmp_path, family):
  # Each family as Hugging Face saves it, against transformers in float32 with random bf16 weights. For gpt-oss,
  # leaving out any one of the sinks, sliding window, YaRN, biases or clamped GLU takes numpy past 0.5%. The chip may
  # pick a token within 0.1% of the span of the best, since Qwen2-MoE has such a tie.
  torch, transformers = pytest.importorskip("torch"), pytest.importorskip("transformers")
  name, extra = FAMILIES[family]
  torch.manual_seed(0)
  config = getattr(transformers, name + "Config")(hidden_size=128, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                                  num_key_value_heads=2, vocab_size=256, max_position_embeddings=64, rope_theta=1e4,
                                                  tie_word_embeddings=False, **extra)
  model = getattr(transformers, name + "ForCausalLM")(config)
  for name, p in model.named_parameters():  # biases start at 0 and norms at 1, so randomize them to make them matter
    if p.ndim == 1 or "bias" in name: p.data.normal_(1, 0.2) if torch.all(p == 1) else p.data.normal_(0, 0.5)
  model = model.to(torch.bfloat16).float()
  model.save_pretrained(tmp_path)
  ids = [3, 1, 4, 1, 5, 9, 2, 6]
  with torch.no_grad(): want = model(torch.tensor([ids])).logits[0].numpy()
  c, weights = llama.load(str(tmp_path), 32)
  rel = lambda got: max(np.abs(got[r] - want[r]).max() / np.abs(want[r]).max() for r in range(len(ids)))
  numpy = rel(llama.forward(ids, list(range(8)), llama.arrays(weights), c, llama.kv_cache(c, 32, np.zeros), every=True))
  assert numpy < 0.005, numpy
  rt = Runtime(chip)
  on = lambda row: Tensor.of(rt, [row], "int32")
  got = gpt2.values(llama.forward(on(ids), on(list(range(8))), llama.upload(rt, weights), c, llama.kv_cache(c, 32, llama.zeros(rt)), every=True))
  ties = [want[r].max() - want[r][got[r].argmax()] <= 0.001 * np.ptp(want[r]) for r in range(len(ids))]  # its best, or one bf16 can't tell from it
  assert rel(got) < 0.02 and all(ties), (rel(got), ties)

def test_qwen3_5_moe(chip, tmp_path, monkeypatch):
  # Qwen3.5-MoE's language model (also Qwen3.6's) from Hugging Face's files, against transformers in float32 with
  # random bf16 weights: a 100-token prompt (two DeltaNet chunks of 64), then replayed decode steps. numpy must be
  # within 0.5% (0.13% and 0.06% measured), and the chip within 3% of numpy (1.2%, 0.8%). At 2 positions of 100 the
  # chip picks a token 0.3% of the span away from numpy's, which bf16 can't tell apart. Tokens use all 8 experts,
  # because with a top 2, some of the 400 choices are near ties that bf16 breaks either way. They're 1e-5 to 1e-3
  # apart with any seed.
  torch, transformers = pytest.importorskip("torch"), pytest.importorskip("transformers")
  if not hasattr(transformers, "Qwen3_5MoeTextConfig"): pytest.skip("needs a transformers with Qwen3.5-MoE")
  # hide fla and causal_conv1d, whose GPU kernels transformers would otherwise use, so it runs its torch code
  monkeypatch.setitem(sys.modules, "fla", None), monkeypatch.setitem(sys.modules, "causal_conv1d", None)
  torch.manual_seed(0)
  config = transformers.Qwen3_5MoeTextConfig(hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                                             linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
                                             num_experts=8, num_experts_per_tok=8, moe_intermediate_size=64, shared_expert_intermediate_size=64,
                                             vocab_size=256, max_position_embeddings=256, layer_types=["linear_attention"] * 3 + ["full_attention"])
  model = transformers.Qwen3_5MoeForCausalLM(config)
  for p in model.parameters():
    if p.ndim == 1: p.data.normal_(0, 0.3) if torch.all(p == 0) else p.data.normal_(1, 0.2)
  model = model.to(torch.bfloat16).float()
  model.save_pretrained(tmp_path)
  rng = np.random.default_rng(0)
  ids, steps = rng.integers(0, 256, 100).tolist(), rng.integers(0, 256, 3).tolist()
  with torch.no_grad():
    out = model(torch.tensor([ids]), use_cache=True)
    want, cache, want_steps = out.logits[0].numpy(), out.past_key_values, []
    for s in steps:
      out = model(torch.tensor([[s]]), past_key_values=cache, use_cache=True)
      want_steps.append(out.logits[0, -1].numpy())
      cache = out.past_key_values
  c, weights = llama.load(str(tmp_path), 128)
  names = {k.split(".", 2)[2] for k in weights if k.startswith("layers.0.")}
  assert c.gated and c.centered and c.partial == 0.25 and {"conv", "decay", "dt_bias"} <= names
  rel = lambda got, want: max(np.abs(g - w).max() / np.abs(w).max() for g, w in zip(got, want))
  w, kv = llama.arrays(weights), llama.kv_cache(c, 128, np.zeros)
  numpy = llama.forward(ids, list(range(100)), w, c, kv, every=True)
  numpy_steps = [llama.forward([s], [100 + i], w, c, kv)[0] for i, s in enumerate(steps)]
  assert rel(numpy, want) < 0.005 and rel(numpy_steps, want_steps) < 0.005, (rel(numpy, want), rel(numpy_steps, want_steps))
  rt = Runtime(chip)
  on = lambda row: Tensor.of(rt, [row], "int32")
  t, ckv = llama.upload(rt, weights), llama.kv_cache(c, 128, llama.zeros(rt))
  got = gpt2.values(llama.forward(on(ids), on(list(range(100))), t, c, ckv, every=True))
  step = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, ckv))
  got_steps = [llama.logits(step(on([s]), on([100 + i]))) for i, s in enumerate(steps)]
  assert rel(got, numpy) < 0.03 and near(got, numpy), rel(got, numpy)
  assert rel(got_steps, numpy_steps) < 0.03 and near(got_steps, numpy_steps), rel(got_steps, numpy_steps)

def test_qwen3_5_sequences(chip):
  # The small Qwen3.5 runs several sequences together, like a server's slots. Each must match the same sequence run
  # alone in numpy within 3% (0.8% measured). A decode step is one replayed call for all the slots. Free slot 1
  # (live 0) must keep its states.
  rng = np.random.default_rng(29)
  c, weights = qwen35(rng)
  w, size, prompts = llama.arrays(weights), 32, {0: [7, 200, 31], 2: [5, 64, 9, 9, 9]}
  p = {k.removeprefix("layers.0."): w[k] for k in w if k.startswith("layers.0.")}
  h, alone = rng.normal(0, 1, (3, 128)), [llama.kv_cache(c, size, np.zeros) for _ in range(3)]
  for s in alone: s[0][1][:] = rng.normal(0, 0.1, s[0][1].shape)  # give each sequence a state of its own
  many = llama.kv_cache(c, 3 * size, np.zeros, 3)
  for u in range(3): many[0][1][4 * u:4 * u + 4] = alone[u][0][1]
  together = qwen.deltanet(h, p, c, many[0], lambda op: None, np.array([[1.0], [0.0], [1.0]]))
  for u in (0, 2): assert np.allclose(together[u], qwen.deltanet(h[u:u + 1], p, c, alone[u][0], lambda op: None)[0], atol=1e-5)
  assert np.array_equal(many[0][1][4:8], alone[1][0][1]), "a free slot's state must not change"
  rt = Runtime(chip)
  t, kv, np_kv = llama.upload(rt, weights), llama.kv_cache(c, 3 * size, llama.zeros(rt), 3), {u: llama.kv_cache(c, size, np.zeros) for u in prompts}
  on, ids = lambda row, d="int32": Tensor.of(rt, [row], d), {u: list(s) for u, s in prompts.items()}
  rel = lambda got, want: np.abs(got - want).max() / np.abs(want).max()
  for u, s in prompts.items():
    got = gpt2.values(llama.forward(on(s), on(list(range(len(s)))), t, c, llama.slot(c, kv, u, 3), every=True))[len(s) - 1]
    want = llama.forward(s, list(range(len(s))), w, c, np_kv[u])[0]
    assert rel(got, want) < 0.03 and near([got], [want]), (u, rel(got, want))
    ids[u].append(int(want.argmax()))
  step = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv, 3))
  for i in range(4):
    places = [len(ids[u]) - 1 if u in ids else size - 1 for u in range(3)]
    got = gpt2.values(step(on([ids[u][-1] if u in ids else 0 for u in range(3)]), on(places)))
    for u in prompts:
      want = llama.forward(ids[u][-1:], [places[u]], w, c, np_kv[u])[0]
      assert rel(got[u], want) < 0.03 and near([got[u]], [want]), (i, u, rel(got[u], want))
      ids[u].append(int(want.argmax()))
  free = llama.slot(c, kv, 1, 3)
  assert not any(np.any(np.frombuffer(s.tobytes(), np.float32)) for l in range(3) for s in free[l]), "the free slot's states stay 0"

def test_delta_step(chip):
  # qwen.delta runs Gated DeltaNet decode in one kernel, with a work unit per key head. It must match numpy_delta to 1e-5 with the
  # conv inputs exact. The shapes go up to Qwen3.6's 32 heads of 128 by 128. The second sequence is inactive (live 0). Its
  # conv and states stay as they were, and where reads are counted, it reads only its live and accept. With m=2 it's a
  # verify step, which starts from the states after both tokens of the last step or after the first, as accept says.
  rt, rng = Runtime(chip), np.random.default_rng(1)
  up = lambda a: Tensor.frombytes(rt, a.shape[-2], a.shape[-1], np.float32(a).tobytes(), a.shape[0], "fp32")
  for m in (1, 2):
    for n, kh, vh, kd, vd in ((1, 2, 4, 32, 32), (3, 2, 4, 64, 96), (4, 16, 32, 128, 128)):
      width = 2 * kh * kd + vh * vd
      x = np.float32(rng.normal(0, 1, (1, n * m, width + vh * vd + 2 * vh)))
      conv, conv2 = (np.float32(rng.normal(0, 1, (3, n, width))) for _ in range(2))
      taps, dt = np.float32(rng.normal(0, 0.5, (4, 1, width))), np.float32(rng.normal(0, 1, (1, 1, vh)))
      decay = np.float32(-rng.uniform(0.5, 2, (1, 1, vh)))
      S, first = (np.float32(rng.normal(0, 0.1, (n * vh, kd, vd))) for _ in range(2))
      live, accept = np.float32(np.arange(n * m) // m != 1).reshape(1, n * m, 1), np.float32(np.arange(n) % 2).reshape(1, 1, n)
      ts = {k: up(a) for k, a in dict(S=S, x=x, conv=conv, taps=taps, dt=dt, decay=decay, first=first, accept=accept, conv2=conv2).items()}
      call = lambda live: qwen.delta(*(ts[k] for k in ("S", "x", "conv", "taps", "dt", "decay")), Tensor.of(rt, live[0].tolist()),
                                     *((ts["first"], ts["accept"], ts["conv2"]) if m > 1 else ()))
      got = value(call(live))
      want_S, want_first, want_conv, want_conv2 = np.float64(S), np.float64(first), np.float64(conv), np.float64(conv2)
      want = qwen.numpy_delta(want_S, x, want_conv, taps, dt, decay, live, *((want_first, accept, want_conv2) if m > 1 else ()))
      for a, b in ((got, want), (value(ts["S"]), want_S)) + (((value(ts["first"]), want_first),) if m > 1 else ()):
        assert np.abs(a - b).max() < 1e-5 * np.abs(b).max(), (m, n, kd, vd, np.abs(a - b).max() / np.abs(b).max())
      for a, b in ((value(ts["conv"]), want_conv),) + (((value(ts["conv2"]), want_conv2),) if m > 1 else ()):
        assert np.array_equal(a, np.float32(b)), (m, n, "the conv's inputs")
      if n == 1: continue
      assert np.array_equal(value(ts["S"])[vh:2 * vh], S[vh:2 * vh]) and np.array_equal(value(ts["conv"])[:, 1], conv[:, 1]), \
        "inactive sequence state must remain unchanged"
      if chip.pci.counts:  # the same call again with no sequence dead, then with the second dead
        reads = []
        for alive in (np.ones_like(live), live):
          since = rt.launches
          call(alive)
          reads += [r[3] for r in rt.profile(since) if r[0] == "delta"]
        assert n * (reads[1] - kh * 64 * m) == (n - 1) * reads[0], (m, n, reads)

def test_delta_pieces(chip):
  # qwen.chunk runs a Gated DeltaNet prompt in pieces of 64 tokens, and must match a piece at a time to 1e-6. Then the
  # small Qwen3.5 runs a 150-token prompt, three pieces with the last one padded, within 5% of numpy.
  rng = np.random.default_rng(47)
  c, weights = qwen35(rng, places=256)
  w = llama.arrays(weights)
  q, k = rng.normal(0, 0.3, (4, 150, 32)), rng.normal(0, 0.3, (4, 150, 32))
  k /= np.linalg.norm(k, axis=-1, keepdims=True)
  v, beta, g, S = rng.normal(0, 1, (4, 150, 48)), rng.uniform(0, 1, (4, 150, 1)), -rng.uniform(0, 0.2, (4, 150, 1)), rng.normal(0, 0.1, (4, 32, 48))
  o, after = qwen.chunk(q, k, v, beta, g, S)
  for s0 in range(0, 150, 64):  # the same pieces as chunk's, in separate calls
    piece, S = qwen.chunk(*(t[:, s0:s0 + 64, :] for t in (q, k, v, beta, g)), S)
    assert np.abs(piece - o[:, s0:s0 + 64, :]).max() < 1e-6
  assert np.abs(after - S).max() < 1e-6
  rt = Runtime(chip)
  t, on = llama.upload(rt, weights), lambda row: Tensor.of(rt, [row], "int32")
  ids = [int(i) for i in rng.integers(0, 256, 150)]
  want = llama.forward(ids, list(range(150)), w, c, llama.kv_cache(c, 256, np.zeros), every=True)
  got = gpt2.values(llama.forward(on(ids), on(list(range(150))), t, c, llama.kv_cache(c, 256, llama.zeros(rt)), every=True))
  assert max(np.abs(a - b).max() / np.abs(b).max() for a, b in zip(got, want)) < 0.05 and near(got, want)

def test_speculative(chip):
  # Two slots verify their token and draft in one replayed call, with a third slot free. A right draft is accepted. A
  # wrong one is rejected (accept 0), and the next step goes on from the Gated DeltaNet states as they were after the
  # first token. The multi-token prediction layer (qwen.draft) must match numpy too, with the output layer and with a
  # bfp4 copy of it.
  rng = np.random.default_rng(37)
  c, weights = qwen35(rng)
  w, size = llama.arrays(weights), 32
  rt = Runtime(chip)
  t, kv = llama.upload(rt, weights), llama.kv_cache(c, 3 * size, llama.zeros(rt), 3, speculative=True)
  on, rel = lambda row, d="int32": Tensor.of(rt, [row], d), lambda got, want: np.abs(got - want).max() / np.abs(want).max()
  ref = lambda s: llama.forward(s, list(range(len(s))), w, c, llama.kv_cache(c, size, np.zeros), every=True)  # numpy, from an empty cache
  seqs = {0: [7, 200, 31], 1: [5, 64, 9, 9, 9]}  # each starts as a prompt and gains the tokens it knows
  for u, s in seqs.items():
    got, h = llama.forward(on(s), on(list(range(len(s)))), t, c, llama.slot(c, kv, u, 3), hidden=True)
    want = ref(s)
    assert rel(gpt2.values(got)[-1], want[-1]) < 0.03
    s.append(int(want[-1].argmax()))
    if u == 0:  # the multi-token prediction layer over the prompt, from each position's hidden state and the token after it
      _, nh = llama.forward(s[:-1], list(range(len(s) - 1)), w, c, llama.kv_cache(c, size, np.zeros), hidden=True)
      mine = gpt2.values(qwen.draft(on(s[1:]), on(list(range(len(s) - 1))), h, t, c, llama.slot(c, kv, u, 3)))
      theirs = qwen.draft(s[1:], list(range(len(s) - 1)), nh, w, c, llama.kv_cache(c, size, np.zeros))
      assert max(rel(mine[r], theirs[r]) for r in range(len(s) - 1)) < 0.03
      c.precision = {("draft head", None): ("bfp4", 1)}
      head = llama.upload(rt, {k: weights[k] for k in ("embed.T", "mtp.fc")}, c.precision)["embed.T draft"]
      assert head.dtype == "bfp4"
      mine = gpt2.values(qwen.draft(on(s[1:]), on(list(range(len(s) - 1))), h, t | {"embed.T draft": head}, c, llama.slot(c, kv, u, 3)))
      theirs = qwen.draft(s[1:], list(range(len(s) - 1)), nh, llama.arrays(weights, c.precision), c, llama.kv_cache(c, size, np.zeros))
      assert max(rel(mine[r], theirs[r]) for r in range(len(s) - 1)) < 0.03
      c.precision = None
  step = tensor.jit(lambda ids, places, accept: llama.forward(ids, places, t, c, kv, 3, accept=accept))
  accept = [1.0, 1.0, 1.0]
  for i in range(6):
    right = {u: int(ref(s)[-1].argmax()) for u, s in seqs.items()}
    drafts = {u: right[u] if (i + u) % 2 == 0 else (right[u] + 1) % 256 for u in seqs}
    ids = [x for u in range(3) for x in ([seqs[u][-1], drafts[u]] if u in seqs else [0, 0])]
    places = [p for u in range(3) for p in ([len(seqs[u]) - 1, len(seqs[u])] if u in seqs else [size - 1, size - 1])]
    got = gpt2.values(step(on(ids), on(places), on(accept, "fp32")))
    for u, s in seqs.items():
      want = ref(s + [drafts[u]])[-2:]
      assert rel(got[2 * u], want[0]) < 0.03 and rel(got[2 * u + 1], want[1]) < 0.03, (i, u, rel(got[2 * u], want[0]), rel(got[2 * u + 1], want[1]))
      accept[u] = float(drafts[u] == right[u])
      s += [drafts[u], int(want[1].argmax())] if accept[u] else [right[u]]

def test_mxfp4(tmp_path):
  # gpt-oss keeps its experts in MXFP4. qwen.mxfp4 must unpack them to bf16 bit for bit like transformers does, and
  # llama.load must split their interleaved gate and up columns apart.
  torch, packed = pytest.importorskip("torch"), pytest.importorskip("transformers.integrations.mxfp4")
  st = pytest.importorskip("safetensors.numpy")
  rng = np.random.default_rng(43)
  blocks, scales = rng.integers(0, 256, (2, 64, 3, 16), dtype=np.uint8), rng.integers(118, 136, (2, 64, 3), dtype=np.uint8)
  want = packed.convert_moe_packed_tensors(torch.from_numpy(blocks), torch.from_numpy(scales)).view(torch.int16).numpy().view(np.uint16)
  assert np.array_equal(qwen.mxfp4(blocks, scales), want)
  config = {"hidden_size": 96, "num_attention_heads": 3, "num_key_value_heads": 1, "num_hidden_layers": 1, "rms_norm_eps": 1e-5,
            "rope_theta": 1e4, "max_position_embeddings": 32, "vocab_size": 32, "intermediate_size": 32, "head_dim": 32,
            "num_experts_per_tok": 1, "model_type": "gpt_oss"}
  (tmp_path / "config.json").write_text(json.dumps(config))
  f32 = lambda *shape: rng.normal(0, 1, shape).astype(np.float32)
  shapes = (("q", (96, 96)), ("k", (32, 96)), ("v", (32, 96)), ("o", (96, 96)))
  attention = {f"model.layers.0.self_attn.{n}_proj.weight": f32(*shape) for n, shape in shapes}
  st.save_file(attention | {"model.embed_tokens.weight": f32(32, 96), "model.norm.weight": f32(96), "lm_head.weight": f32(32, 96),
                "model.layers.0.mlp.experts.gate_up_proj_blocks": blocks, "model.layers.0.mlp.experts.gate_up_proj_scales": scales,
                "model.layers.0.mlp.experts.gate_up_proj_bias": f32(2, 64), "model.layers.0.mlp.experts.down_proj": f32(2, 32, 96),
                "model.layers.0.mlp.experts.down_proj_bias": f32(2, 96), "model.layers.0.mlp.router.weight": f32(2, 96),
                "model.layers.0.mlp.router.bias": f32(2)},
               str(tmp_path / "model.safetensors"))
  c, weights = llama.load(str(tmp_path), 32)
  assert np.array_equal(weights["layers.0.experts.gate_up"].bits(), np.concatenate([want[..., ::2], want[..., 1::2]], -1)) and c.limit == 7.0

TWO_CHIPS = r"""
import os
from baretile import bh, pci
from baretile.chip import Chip
from baretile.runtime import Runtime
from baretile.tensor import Tensor
chips = [Chip(pci.open_device(n)) for n in range(2)]
rts = [Runtime(chip) for chip in chips]
for n, rt in enumerate(rts): assert (Tensor.of(rt, [[1.0, 2.0]]) * (n + 2)).tolist() == [[n + 2.0, 2 * n + 4.0]], n
(a, links_a), (b, links_b) = (chip.links() for chip in chips)
assert a != b and len(links_a) == len(links_b) == 2, (a, links_a, b, links_b)
assert all(peer == b and links_b[back] == (a, ch) for ch, (peer, back) in links_a.items()), (links_a, links_b)
for src, dst, links in ((chips[0], chips[1], links_a), (chips[1], chips[0], links_b)):
  for ch, (_, back) in links.items():
    data, flag = os.urandom(8192), os.urandom(16)
    src.write(*bh.ETH[ch], 0x20000, data + flag)
    dst.write(*bh.ETH[back], 0x40000, bytes(len(data) + 16))
    src.send(ch, 0x20000, 0x40000, len(data))
    src.send(ch, 0x20000 + len(data), 0x40000 + len(data), 16)
    src.wait(lambda: dst.read(*bh.ETH[back], 0x40000 + len(data), 16) == flag, "deliver the flag", 2.0)
    assert dst.read(*bh.ETH[back], 0x40000, len(data)) == data, ch
import random
x = [[[random.uniform(-4, 4) for _ in range(100)] for _ in range(70)] for _ in range(2)]
t = Tensor.of(rts[0], x)  # 24 tiles: 6 chunks through the 2 slots, so the credits count
for ch, (_, back) in links_a.items():
  y = t.to(rts[1], ch, back)
  assert y.rt is rts[1] and y.tolist() == t.tolist() and (y * 2).to(rts[0], back, ch).tolist() == (t * 2).tolist(), ch
import sys, numpy as np  # a model split across the two chips: its first block on chip 0, its second and its head on chip 1
from types import SimpleNamespace
sys.path.insert(0, "models")
import gpt2, llama
c, rng = llama.model(dim=128, heads=4, kv=2, layers=2, eps=1e-5, theta=1e4), np.random.default_rng(3)
draw = lambda mean, std, shape: gpt2.bf16_floats(gpt2.bf16_bits(rng.normal(mean, std, shape)))
w = {f"layers.{l}.{k}": draw(*how) for l in range(2) for k, how in (("input_layernorm", (1, 0.1, (1, 128))), ("qkv_proj", (0, 0.05, (128, 256))),
     ("o_proj", (0, 0.05, (128, 128))), ("post_attention_layernorm", (1, 0.1, (1, 128))), ("gate_up_proj", (0, 0.05, (128, 512))),
     ("down_proj", (0, 0.05, (256, 128))))}
w |= {"norm": draw(1, 0.1, (1, 128)), "embed": draw(0, 1, (256, 128))}
w |= {"embed.T": w["embed"].T.copy()} | dict(zip(("cos", "sin", "turn"), llama.rotary(c, 32)))
ids = [5, 17, 3, 200, 9]
want = llama.forward(ids, list(range(5)), w, c, llama.kv_cache(c, 32, np.zeros))[0]
up = [{k: Tensor.of(rt, v.tolist()) for k, v in w.items()} for rt in rts]
places = [Tensor.of(rt, [list(range(5))], "int32") for rt in rts]
caches = [llama.kv_cache(c, 32, llama.zeros(rt)) for rt in rts]
f = llama.fns(up[0]["norm"])
x = f.rows(up[0]["embed"], Tensor.of(rts[0], [ids], "int32"))
ch, (_, back) = next(iter(links_a.items()))
for l, n in ((0, 0), (1, 1)):
  if x.rt is not rts[n]: x = x.to(rts[n], ch, back)  # the hidden state crosses the link
  rotary = (f.rows(up[n]["cos"], places[n]), f.rows(up[n]["sin"], places[n]), up[n]["turn"])
  p = gpt2.block(up[n], f"layers.{l}.")
  x = llama.layer(x, p, c, rotary, gpt2.mask(places[n], 32), caches[n][l], places[n], l, True)
got = llama.logits(f.mm(f.rows(llama.rms(x, up[1]["norm"], c), [4]), up[1]["embed.T"], None))
assert np.abs(got - want).max() < 0.02 * np.ptp(want) and got.argmax() == want.argmax(), (np.abs(got - want).max(), np.ptp(want))
print("ok")
"""

@pytest.mark.skipif("TTSIM2" not in os.environ, reason="TTSIM2=path/to/libttsim.so of ttsim's two-chip build (bh_x2) runs it")
def test_two_chips():
  # ttsim's two-chip build is a P300: two Blackholes joined by two Ethernet links. It runs in its own process, since a
  # process can load only one libttsim. Chip.links must pair the links, Chip.send and Tensor.to must move data across
  # each link in both directions, and a model split across the two chips must give numpy's logits.
  import subprocess
  run = subprocess.run([sys.executable, "-c", TWO_CHIPS], env=os.environ | {"TTSIM": os.environ["TTSIM2"]}, capture_output=True, text=True,
                       timeout=600, cwd=pathlib.Path(__file__).parent.parent)
  assert run.returncode == 0 and run.stdout.strip().endswith("ok"), run.stdout + run.stderr

@pytest.mark.skipif("LLAMA8B" not in os.environ, reason="LLAMA8B=1 runs it: about four minutes on ttsim")
def test_llama_8b_shapes(chip):
  # One block in Llama-3.1-8B's shapes at TT's performance precision, so B2's matmuls, casts and attention run at
  # full size. Only the vocabulary is cut, to 2048. On ttsim the casts in llama.upload take minutes, and upload waits
  # for them. The MLP's 4096 x 28672 in bfp4 is 114,688 tiles.
  rt, rng = Runtime(chip), np.random.default_rng(8)
  rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  c = llama.model(dim=4096, heads=32, kv=8, layers=1, eps=1e-5, theta=500000.0, rope=rope, vocab=2048, ffn=14336, places=96)
  bits = lambda shape, scale, mean=0.0: gpt2.bf16_bits(rng.normal(mean, scale, shape))
  w = {"layers.0.input_layernorm": bits((1, 4096), 0.1, 1), "layers.0.post_attention_layernorm": bits((1, 4096), 0.1, 1),
       "layers.0.qkv_proj": bits((4096, 6144), 0.02), "layers.0.o_proj": bits((4096, 4096), 0.02), "layers.0.gate_up_proj": bits((4096, 28672), 0.02),
       "layers.0.down_proj": bits((14336, 4096), 0.02), "norm": bits((1, 4096), 0.1, 1), "embed": bits((2048, 4096), 1.0)}
  w |= {"embed.T": np.ascontiguousarray(w["embed"].T)} | {k: gpt2.bf16_bits(a) for k, a in zip(("cos", "sin", "turn"), llama.rotary(c, c.places))}
  weights, c.precision = {k: llama.Weight(lambda v=v: v) for k, v in w.items()}, llama.precision(c, "tt")
  t = llama.upload(rt, weights, c.precision)
  arrays, on = llama.arrays(weights, c.precision), lambda row: Tensor.of(rt, [row], "int32")
  kv, np_kv = llama.kv_cache(c, 96, llama.zeros(rt, "bfp8")), llama.kv_cache(c, 96, np.zeros)
  prompt = [int(i) for i in rng.integers(0, 2048, 64)]
  for new, places in ((prompt, list(range(64))), ([5], [64]), ([9], [65])):
    out = llama.forward(on(new), on(places), t, c, kv).realize()
    rt.wait(600)
    got, want = llama.logits(out), llama.forward(new, places, arrays, c, np_kv)[0]
    assert got.argmax() == want.argmax() and np.abs(got - want).max() < 0.02 * np.ptp(want), (np.abs(got - want).max(), np.ptp(want))

def test_benchmark_table():
  # The speed column is baretile's over TT-NN's, inverted for times, so above 1 means baretile is faster. A matmul at
  # the card's peak rate has a roofline of 1.
  ours = {"versions": {"lines of code": {"baretile": 1}}, "clock": "the host's", "B1 matmul 4096^3 bf16, 2 phases": {"ms": 3.0},
          "B2 decode, batch 1": {"tokens/s": 60.0}, "B3 prefill, 32768 tokens": {"does not run": "MemoryError"}}
  theirs = {"versions": {"lines of code": {"tt-metal": 2}}, "clock": "the host's", "B1 matmul 4096^3 bf16, 2 phases": {"ms": 1.5},
            "B2 decode, batch 1": {"tokens/s": 50.0}, "B3 prefill, 32768 tokens": {"seconds to first token": 9.0}}
  rows = benchmark.table(ours, theirs).splitlines()
  assert rows[2].endswith("| 0.50 (not above 1) |") and rows[3].endswith("| 1.20 (above 1) |") and rows[4].endswith("|  |"), rows
  assert all(r.count("|") == 5 for r in rows) and "MemoryError" in rows[4], rows
  assert benchmark.roofline(cost.CARD["tflops"] / 2, 1.0, 2) == 1
  c = llama.model(layers=2, dim=256, heads=4, kv=2, ffn=512, vocab=1000)
  attention = 2 * 4 * 4 * llama.head(c) * 10 * 11 // 2  # two blocks, scores and values, 4 heads, causal over 10 tokens
  assert benchmark.prefill_flops(c, 10) == 2 * 10 * (cost.shape(c, "tt").values - 1000 * 256) + 2 * 1000 * 256 + attention

def test_benchmark_load(chip, tmp_path):
  # B4's load against serve.py for a small model: two agents, two streamed turns each. The second turn extends the
  # first, so most of its prompt should already be cached.
  rt, rng = Runtime(chip), np.random.default_rng(37)
  c, w = smol(rng, 2, places=128)
  engine = serve.Engine(rt, {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, c, bytes_tokenizer(tmp_path), slots=2, places=128, chunk=32)
  threading.Thread(target=engine.run, daemon=True).start()
  httpd = serve.server(engine, "smol", port=0)
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  got = benchmark.load(f"http://localhost:{httpd.server_address[1]}", 2, SimpleNamespace(system=6, task=2, tool=2, out=4, n=2),
                       " ".join(f"w{i}" for i in range(40)))
  httpd.shutdown()
  assert got["turns"] == 4 and got["tokens/s"] > 0 and got["seconds to first token"] > 0 and got["prompt tokens cached"] > 0.3, got

def test_chunked_prefill(chip):
  # llama.prefill runs a long prompt a chunk at a time, so attention's scores are a chunk by the cache instead of the
  # prompt squared. B3's 32,768 tokens need this. Here 70 tokens go in chunks of 32, 32 and 6.
  rt, rng = Runtime(chip), np.random.default_rng(29)
  c, w = smol(rng, 2, places=96)
  t, prompt = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, [int(i) for i in rng.integers(0, 256, 70)]
  on = lambda row: Tensor.of(rt, [row], "int32")
  kv, np_kv = llama.kv_cache(c, 96, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 96, np.zeros)
  want = llama.prefill(prompt, w, c, np_kv, 32)[0]
  assert np.allclose(want, llama.forward(prompt, range(70), w, c, llama.kv_cache(c, 96, np.zeros))[0], atol=1e-3)
  got = llama.logits(llama.prefill(prompt, t, c, kv, 32, on))
  assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), np.abs(got - want).max()
  nxt = int(want.argmax())
  got, want = llama.logits(llama.forward(on([nxt]), on([70]), t, c, kv)), llama.forward([nxt], [70], w, c, np_kv)[0]
  assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), np.abs(got - want).max()

def test_llama_precision(chip):
  # TT's performance precision puts the MLP's gate and up in bfp4 and the rest in bfp8, except in Llama-3.1-8B's last
  # block. numpy with the same block-float values but exact products must pick the same tokens, within 1% of the span.
  # upload waits for each cast, so the bf16 copy of the next weight reuses the same DRAM.
  big = llama.model(layers=32, dim=4096)
  assert llama.precision(big, "tt")["gate_up", 31] == ("bfp8", 2) and llama.precision(big, "tt")["gate_up", 30] == ("bfp4", 1)
  assert llama.precision(big, "tt")["head", None] == ("bfp8", 2) and llama.precision(big, "bf16")["qkv", 0] == ("bf16", 4)
  tt = llama.precision(big, "tt")
  assert [tt[k] for k in (("attention", "decode"), ("attention", "prefill"), ("kv", None))] == [(None, 2), (None, 4), ("bfp8", None)]
  # "experts" fits Qwen3.6 on one card. It puts the MLP matrices in bfp4 at one phase, the rest in bfp8 at two, and a
  # draft's output layer in bfp4
  experts = llama.precision(big, "experts")
  keys = (("gate_up", 31), ("down", 0), ("qkv", 5), ("head", None), ("draft head", None), ("kv", None))
  assert [experts[k] for k in keys] == [("bfp4", 1), ("bfp4", 1), ("bfp8", 2), ("bfp8", 2), ("bfp4", 1), ("bfp8", None)]
  assert llama.weight_format("layers.3.shared_expert.down_proj", experts) == llama.weight_format("layers.3.experts.down", experts) == "bfp4"
  assert llama.weight_format("embed.T draft", experts) == "bfp4" and ("draft head", None) not in llama.precision(big, "tt")
  rt, rng = Runtime(chip), np.random.default_rng(5)
  c, w = smol(rng, 2)
  c.precision = llama.precision(c, "tt")
  weights = {k: llama.Weight(lambda v=v: (np.float32(v).view(np.uint32) >> 16).astype(np.uint16)) for k, v in w.items()}
  t = llama.upload(rt, weights, c.precision)
  rt.wait()  # the DRAM that upload freed comes back
  reached, held = max(a + n for a, n in rt.used.items()), sum(rt.used.values())  # bytes in each DRAM bank
  assert reached - held <= -(-max(x.tiles for x in t.values()) // 8) * 2048, "the DRAM of a cast weight's bf16 copy wasn't reused"
  formats = {k: x.dtype for k, x in t.items() if "proj" in k or k == "embed.T"}
  assert formats == {k: "bfp4" if "gate_up" in k else "bfp8" for k in t if "proj" in k or k == "embed.T"}
  sizes = SimpleNamespace(**vars(c), vocab=256, places=64, ffn=1536)  # benchmark.weights, from the sizes alone, must match upload
  assert {k: (x.rows, x.cols, x.dtype) for k, x in t.items()} == {k: (x.rows, x.cols, x.dtype) for k, x in benchmark.weights(sizes).items()}
  arrays = llama.arrays(weights, c.precision)  # numpy rounds to the formats the way the chip casts (test_bfp)
  kv, np_kv = llama.kv_cache(c, 32, llama.zeros(rt, "bfp8")), llama.kv_cache(c, 32, np.zeros)  # numpy's cache rounds like a bfp8 one
  assert kv[0].dtype == "bfp8"
  on = lambda row: Tensor.of(rt, [row], "int32")
  for ids, places in (([7, 200, 31], [0, 1, 2]), ([5], [3])):  # the prompt, then a token decoded
    got, want = llama.logits(llama.forward(on(ids), on(places), t, c, kv)), llama.forward(ids, places, arrays, c, np_kv)[0]
    assert got.argmax() == want.argmax() and np.abs(got - want).max() < 0.01 * np.ptp(want), (np.abs(got - want).max(), np.ptp(want))

def test_token_accuracy(chip):
  # llama.accuracy is Tenstorrent's token accuracy, which B2 uses. Each step is fed the reference's next token, so the
  # model's picks don't steer the run. A reference that's wrong at one position costs top-1 that step.
  rt, rng = Runtime(chip), np.random.default_rng(6)
  c, w = smol(rng, 2)
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  tokens = [int(i) for i in rng.integers(0, 256, 44)]
  prompt, after, top5, kv = tokens[:36], tokens[36:], [], llama.kv_cache(c, 64, np.zeros)
  for i in range(8):
    ids, places = (prompt, range(36)) if i == 0 else ([after[i - 1]], [35 + i])
    top5.append([int(k) for k in np.argsort(-llama.forward(ids, places, w, c, kv)[0])[:5]])
  assert llama.accuracy(rt, t, c, prompt, after, top5, 8)[:2] == (1, 1)
  top5[3] = top5[3][1:] + top5[3][:1]  # move its most likely to last, so it's still in the top 5
  top1, five, picks = llama.accuracy(rt, t, c, prompt, after, top5, 8)
  assert (top1, five) == (7 / 8, 1) and picks[3] == top5[3][-1], (top1, five, picks)
  assert llama.accuracy(None, w, c, prompt, after, top5, 8) == (top1, five, picks), "numpy (rt None) should give the same picks and scores"

def test_benchmark_quick_accuracy():
  # --quick decodes after a short prompt to save time, but token accuracy has to be scored from the prompt the
  # reference saw, since its top five came from that context. Scored against its own top five, a model gets 8 of 8.
  rng = np.random.default_rng(6)
  c, w = smol(rng, 2, places=96)
  w |= {k: v * 100 for k, v in w.items() if k.endswith("o_proj")}  # so attention, and with it the context, moves the logits
  tokens = [int(i) for i in rng.integers(0, 256, 76)]
  prompt, after = tokens[:68], tokens[68:]
  logits = llama.forward(prompt + after[:7], list(range(75)), w, c, llama.kv_cache(c, 96, np.zeros), every=True)[67:]
  top5 = [[int(k) for k in np.argsort(-row)[:5]] for row in logits]
  short, scored = benchmark.prompts(prompt, benchmark.SIZES[True])
  assert len(short) == 32 and llama.accuracy(None, w, c, scored, after, top5, 8)[:2] == (1, 1)
  assert llama.accuracy(None, w, c, short, after, top5, 8)[:2] != (1, 1), "the short prompt should score worse"

def test_benchmark_bytes():
  # The bytes a decode token reads, which B2's effective DRAM bandwidth comes from. Llama-3.1-8B reads 6.20 GB a token,
  # so 41.3 tokens/s is half the card's 512 GB/s.
  c = llama.model(dim=4096, heads=32, kv=8, layers=32, vocab=128256, ffn=14336, places=736)
  c.precision = llama.precision(c, "tt")
  t = benchmark.weights(c)
  names = ("layers.0.gate_up_proj", "layers.31.gate_up_proj", "layers.0.down_proj", "embed.T", "embed")
  assert [t[k].dtype for k in names] == ["bfp4", "bfp8", "bfp8", "bfp8", "bf16"]
  read = benchmark.bytes_per_token(t, c, 612.5)
  assert round(read / 1e9, 2) == 6.20 and round(41.3 * read / 512e9, 3) == 0.5, read

def test_batched_decode(chip):
  # Four sequences with prompts of different lengths decode together, each checked against numpy decoding it alone.
  # Their KV caches sit side by side in a shared one, and tokens read only their own rows.
  rt, rng = Runtime(chip), np.random.default_rng(19)
  c, w = smol(rng, 2)
  t, size, prompts = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, 32, [[7, 200, 31], [5, 64], [9, 9, 9, 9, 9], [100]]
  kv = llama.kv_cache(c, size * len(prompts), lambda shape: Tensor.of(rt, np.zeros(shape).tolist()))
  np_kv = [llama.kv_cache(c, size, np.zeros) for _ in prompts]
  on = lambda row: Tensor.of(rt, [row], "int32")
  step, ids = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv, len(prompts))), [list(p) for p in prompts]
  for u, p in enumerate(prompts):  # prefill each prompt into its slot's rows. Its last token's logits give the next token
    got = gpt2.values(llama.forward(on(p), on(list(range(len(p)))), t, c, llama.slot(c, kv, u, len(prompts))))[0]
    want = llama.forward(p, range(len(p)), w, c, np_kv[u])[0]
    assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), (u, np.abs(got - want).max())
    ids[u].append(int(want.argmax()))
  for i in range(4):
    places = [len(s) - 1 for s in ids]
    got = gpt2.values(step(on([s[-1] for s in ids]), on(places)))
    for u, s in enumerate(ids):
      want = llama.forward(s[-1:], [places[u]], w, c, np_kv[u])[0]
      assert np.abs(got[u] - want).max() < 0.03 * np.abs(want).max(), (i, u, np.abs(got[u] - want).max())
      s.append(int(want.argmax()))

def test_serve_timeout(tmp_path):
  # A request whose text doesn't come within `wait` seconds gets a 504 and gives up its slot, so a hung chip doesn't
  # hold slots for requests nobody is waiting on. The stand-in engine takes requests and never answers.
  import queue, urllib.error, urllib.request
  class Hung:
    tok, places, waiting, room = bytes_tokenizer(tmp_path), 64, queue.Queue(), threading.BoundedSemaphore(1)
    offer = serve.Engine.offer
  hung = Hung()
  http = serve.server(hung, "hung", port=0, wait=0.2)
  threading.Thread(target=http.serve_forever, daemon=True).start()
  with pytest.raises(urllib.error.HTTPError) as e:
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{http.server_address[1]}/v1/completions",
                                                  json.dumps({"prompt": "Hi"}).encode(), {"Content-Type": "application/json"}), timeout=60)
  http.shutdown()
  assert e.value.code == 504 and hung.waiting.get_nowait().gone, "the engine should drop a request that timed out"

def bytes_tokenizer(folder):  # writes a byte-level vocabulary of the 256 bytes to tokenizer.json in folder
  bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
  chars = map(chr, bs + [256 + i for i in range(256 - len(bs))])
  vocab = dict(zip(chars, bs + [b for b in range(256) if b not in bs]))
  (folder / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": []}))
  return llama.Tokenizer(str(folder))

def test_heads(chip):
  # serve.heads finds each row's 64 largest logits on the chip, in numpy's stable argsort order, and the log of the sum
  # of exp(x t) over the row. Qwen3.6's vocabulary splits into 97 parts of 2,560. Llama 3's doesn't split into parts of
  # 96 tiles or fewer, so it takes 24 parts of 5,344, as big as topk's L1 holds. serve.Head.of must agree on the host.
  rt, rng = Runtime(chip), np.random.default_rng(3)
  x = np.float32(rng.integers(-3, 4, (40, 300))) * np.float32(rng.choice([1, 0.5], (40, 300)))  # many ties, -0 among them
  x[x == 0] = np.where(rng.random((x == 0).sum()) < 0.5, -0.0, 0.0)
  assert all(np.array_equal(serve.Head.of(row, 1.0, k).ids, np.argsort(-row, kind="stable")[:k]) for row, k in zip(x, range(1, 81, 2)))
  for vocab in (512, 128256, 248320):
    x, ts = rng.normal(0, 3, (3, vocab)).astype(np.float32), [1.0, 1 / 0.7, 2.0]
    x[1, 5] = x[1, vocab - 7] = x[1].max() + 1
    logits = Tensor.frombytes(rt, 3, vocab, x.tobytes(), dtype="fp32")
    values, ids, lse = serve.heads(logits, Tensor.of(rt, [[t] for t in ts], "fp32"))
    want, scaled = np.argsort(-x, -1, kind="stable")[:, :64], np.float64(x) * np.array(ts)[:, None]
    assert np.array_equal(gpt2.values(ids), want) and np.array_equal(gpt2.values(values), np.take_along_axis(x, want, -1)), vocab
    exact = scaled.max(1) + np.log(np.exp(scaled - scaled.max(1, keepdims=True)).sum(1))
    assert np.allclose(gpt2.values(lse)[:, 0], exact, rtol=1e-6, atol=0), (gpt2.values(lse)[:, 0], exact)
    assert np.array_equal(gpt2.values(serve.heads(logits, k=1)[1])[:, 0], want[:, 0])

def test_sampling():
  # Request.pick samples from the head of a row (serve.Head), checked against a full sort of the vocabulary. top_p on a
  # flat distribution reaches past the head into the rest of the row. At temperature 0 the penalties match vLLM's,
  # and repetition divides a positive logit.
  rng = np.random.default_rng(5)
  for logits in (rng.normal(0, 2, 5000), rng.normal(0, 1e-3, 5000)):  # the second has a nucleus of more than 1,024 tokens
    p, order, head = np.exp(logits - logits.max()), np.argsort(-logits), serve.Head.of(logits)
    nucleus = set(order[:np.searchsorted(np.cumsum(p[order]) / p.sum(), 0.9) + 1].tolist())
    for keep, kw in ((nucleus, dict(top_p=0.9)), (set(order[:3].tolist()), dict(top_k=3)), (set(np.flatnonzero(p >= 0.2).tolist()), dict(min_p=0.2))):
      r = serve.Request(None, [], seed=1, **kw)
      got = {r.pick(head) for _ in range(3000 if len(keep) < 1000 else 20000)}
      assert got <= keep and (len(got) > 0.9 * len(keep) if len(keep) > 1000 else got == keep), (kw, len(got), len(keep))
  logits = rng.normal(0, 1.5, 400)  # at temperature 0.8, a fifth of the draws fall past the head's 64
  p, r = np.exp((logits - logits.max()) / 0.8), serve.Request(None, [], temperature=0.8, seed=2)
  p, head, tail = p / p.sum(), serve.Head.of(logits, 1 / 0.8), np.argsort(-logits)[64:]
  share = np.bincount([r.pick(head) for _ in range(20000)], minlength=400) / 20000
  assert 0.15 < p[tail].sum() and abs(share[tail].sum() - p[tail].sum()) < 0.015 and np.all(np.abs(share - p) < 5 * np.sqrt(p / 20000) + 1e-3)
  logits = np.array([3.0, 2.5, 1.0, -1.0])
  for made, prompt, kw, want in (([0], [], dict(presence=1.0), 1), ([0, 0], [], dict(frequency=0.4), 1), ([0], [], dict(frequency=0.4), 0),
                                 ([], [0], dict(repetition=1.3), 1), ([], [0], dict(repetition=1.1), 0), ([], [3], dict(repetition=10.0), 0)):
    r = serve.Request(None, prompt, temperature=0, **kw)
    r.made = made
    assert r.pick(serve.Head.of(logits)) == want, (made, prompt, kw)

def test_shapes(tmp_path):
  # Request.pick restricts output to the tool call or JSON schema requested. Logits that favor a greeting must still
  # produce a tool call when required; random logits must produce valid JSON under the schema.
  tok = bytes_tokenizer(tmp_path)
  tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
  call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
  def spell(text, n, choice):  # greedy: each step favors the text's next character (or the end, 0), then "H"
    r = serve.Request(tok, [], n, 0, eos=[0], shape=serve.chosen(choice, tools))
    for i in range(n):
      logits = np.zeros(256, np.float32)
      logits[ord("H")], logits[ord(text[i]) if i < len(text) else 0] = 1, 2
      if r.take(logits, True): break
    return tok.decode(r.made), r.finish
  assert spell("Hi there", 40, None) == ("Hi there", "stop")
  said, finish = spell("Hi there", 40, "required")
  assert said.startswith("<tool_call>\n<function=read>") and finish == "length", said
  assert spell(call, 100, "required") == (call, "stop"), "a complete call may end"
  assert spell(call.replace("read", "rest"), 40, {"type": "function", "function": {"name": "read"}})[0].startswith("<tool_call>\n<function=read>")
  said, finish = spell(call, len(call) + 5, "none")
  assert "<tool_call>" not in said and said.startswith("<tool_callH") and finish == "stop", said
  with pytest.raises(ValueError): serve.chosen({"type": "function", "function": {"name": "write"}}, tools)
  schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "unit": {"enum": ["C", "F"]}, "n": {"type": "integer"},
                                             "tags": {"type": "array", "items": {"$ref": "#/$defs/tag"}, "maxItems": 2},
                                             "note": {"anyOf": [{"const": "hi"}, {"type": "null"}]}},
            "required": ["ok", "unit", "n", "tags", "note"], "additionalProperties": False, "$defs": {"tag": {"enum": ["a", "b"]}}}
  fits = serve.formatted({"type": "json_schema", "json_schema": {"schema": schema}})
  whole = '{"ok": true, "unit": "C", "n": -12, "tags": ["a"], "note": null}'
  assert [fits(whole[:k]) for k in (0, 1, 9, 20, 32, len(whole) - 1, len(whole))] == [1, 1, 1, 1, 1, 1, 2] and fits(whole + "\n") == 2
  for text in ('{"ok": 1', '{"unit": "K"', '{"n": 1.5', '{"n": 1.', '{"n": -e', '{"x"', '{"tags": ["a", "b", "a"]', '{"note": "ho"', whole + "x",
               "```json", "{\n  "):
    assert fits(text) == 0, text
  assert serve.formatted({"type": "json_object"})('{"a": [1, {"b": "c\\n"}]}') == 2 and serve.formatted({"type": "json_object"})("[1]") == 0
  for seed in range(3):  # with random logits most bytes don't fit, so most tokens are drawn again
    rng, r = np.random.default_rng(seed), serve.Request(tok, [], 300, 1.0, eos=[0], seed=seed, shape=fits)
    while not r.take(rng.normal(0, 1, 256).astype(np.float32), True): pass
    got = json.loads(tok.decode(r.made))
    assert r.finish == "stop" and set(got) == set(schema["properties"]) and isinstance(got["ok"], bool) and isinstance(got["n"], int)
    assert got["unit"] in ("C", "F") and set(got["tags"]) <= {"a", "b"} and len(got["tags"]) <= 2 and got["note"] in ("hi", None), got

def test_serve(chip, tmp_path):
  # The serve.py engine, against numpy's greedy tokens. Three requests on two slots decode together. When two run
  # again, they find their slots still holding their prompts, and each prefills one token. A free slot idles on a row
  # no sequence reaches, since any other row would spoil what the slot holds. A long prompt is prefilled a chunk
  # between steps, so the others keep getting tokens. Then the HTTP side. When no slot is free, a request gets an
  # immediate 429. OpenRouter wants that, so it can try another provider.
  import urllib.request
  rt, rng = Runtime(chip), np.random.default_rng(23)
  c, w = smol(rng, 2)
  w = {k: bf16(v * 10) if k.endswith("_proj") else v for k, v in w.items()}  # sharp attention, so a wrong cached row changes the tokens
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  tok = bytes_tokenizer(tmp_path)
  engine = serve.Engine(rt, t, c, tok, slots=2, places=64, chunk=16)
  threading.Thread(target=engine.run, daemon=True).start()
  def done(r):
    while r.out.get(timeout=600) is not None: pass
    return r.made
  prompts = [[7, 200, 31, 64, 5], [5, 64], list(range(40, 60))]  # the last is longer than a chunk
  together = [serve.Request(tok, p, 6, 0) for p in prompts]
  for r in together: engine.waiting.put(r)
  together = [done(r) for r in together]
  def greedy(prompt, n):  # numpy's tokens for the same prompt, as models/llama.py decodes
    kv, ids = llama.kv_cache(c, 64, np.zeros), list(prompt)
    logits = llama.forward(ids, list(range(len(ids))), w, c, kv)[0]
    for i in range(n - 1): ids.append(int(logits.argmax())); logits = llama.forward(ids[-1:], [len(ids) - 1], w, c, kv)[0]
    return ids[len(prompt):] + [int(logits.argmax())]
  assert together == [greedy(p, 6) for p in prompts], "requests must get numpy's greedy tokens while they decode together"
  before = engine.prefilled
  assert [done(r) for r in [serve.Request(tok, p, 6, 0) for p in prompts[:0:-1]] if not engine.waiting.put(r)] == together[:0:-1]
  assert engine.prefilled - before == 2, "only the last token of each prompt should be prefilled, the rest are in their slots"
  class Watched(serve.Request):  # records engine.prefilled at each of its tokens, from the engine's thread
    def take(self, logits, room): self.seen = getattr(self, "seen", []) + [engine.prefilled]; return super().take(logits, room)
  r, deadline = Watched(tok, [3, 1, 4], 12, 0), time.time() + 600
  engine.waiting.put(r)
  while not getattr(r, "seen", None): assert time.time() < deadline; time.sleep(0.01)
  engine.waiting.put(long := serve.Request(tok, list(range(100, 150)), 2, 0))  # 50 tokens: four chunks
  done(r), done(long)
  steps = np.diff(r.seen)
  assert 16 in steps and steps.max() <= 16, ("at most one chunk of the long prompt between two of its tokens", r.seen)
  r = serve.Request(tok, [], 9, 0, stop=["bc"])
  said = lambda: [r.out.get_nowait() for _ in range(r.out.qsize())]
  assert not r.take(np.eye(256)[97], True) and said() == ["a"]
  assert not r.take(np.eye(256)[98], True) and said() == [], "b might begin the stop string"
  assert r.take(np.eye(256)[99], True) and said() == [] and (r.said, r.finish) == ("a", "stop"), "the engine sends its end (None)"
  r = serve.Request(tok, [], 9, 0, eos=[0])
  assert not r.take(np.eye(256)[97], True) and r.take(np.eye(256)[0], True) and (r.made, r.said, r.finish) == ([97], "a", "stop")
  r = serve.Request(tok, [], 9, 0)
  r.gone = True  # its client has gone
  assert r.take(np.eye(256)[97], True) and (r.made, r.finish) == ([], "gone") and r.out.empty()
  http = serve.server(engine, "smol", port=0, defaults={"temperature": 0})
  threading.Thread(target=http.serve_forever, daemon=True).start()
  url = f"http://127.0.0.1:{http.server_address[1]}/v1"
  post = lambda body: urllib.request.urlopen(urllib.request.Request(url + "/completions", json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=600)
  r = serve.Request(tok, tok.encode("Hi there"), 6, 0)
  engine.waiting.put(r)
  text = tok.decode(done(r))
  # held is what the slot already holds. leaving out temperature gets the model's default
  reply, held = json.load(post({"prompt": "Hi there", "max_tokens": 6})), len(tok.encode("Hi there")) - 1
  assert reply["choices"][0]["text"] == text and reply["choices"][0]["finish_reason"] == "length" and reply["usage"]["completion_tokens"] == 6
  assert reply["usage"]["prompt_tokens_details"]["cached_tokens"] == held, "all but the prompt's last token, which runs again"
  events, usage = streamed(post({"prompt": "Hi there", "max_tokens": 6, "temperature": 0, "stream": True}))
  assert "".join(e["text"] for e in events) == text and events[-1]["finish_reason"] == "length" and usage["completion_tokens"] == 6
  schema = {"type": "object", "properties": {"answer": {"enum": ["yes", "no"]}}, "required": ["answer"], "additionalProperties": False}
  shaped = json.load(post({"prompt": "Hi there", "max_tokens": 80, "temperature": 0,
                           "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}))["choices"][0]
  assert json.loads(shaped["text"])["answer"] in ("yes", "no") and shaped["finish_reason"] == "stop", shaped  # the schema allows no more text
  # greedy: each token is the most likely
  chances = json.load(post({"prompt": "Hi there", "max_tokens": 6, "temperature": 0, "logprobs": 2}))["choices"][0]["logprobs"]
  assert len(chances["tokens"]) == 6 and all(p == max(top.values()) <= 0 for p, top in zip(chances["token_logprobs"], chances["top_logprobs"]))
  assert json.load(urllib.request.urlopen(url + "/models", timeout=60))["data"][0]["id"] == "smol"
  # two streams take both slots, and each holds its slot from the headers on
  holding = [post({"prompt": "Hi there", "max_tokens": 20, "temperature": 0, "stream": True}) for _ in range(2)]
  with pytest.raises(urllib.error.HTTPError) as e: post({"prompt": "Hi", "max_tokens": 2})
  assert e.value.code == 429 and json.load(e.value)["error"]["code"] == 429, "with no slot free, expected an immediate 429, not a wait"
  for h in holding: streamed(h)
  assert json.load(post({"prompt": "Hi", "max_tokens": 2}))["usage"]["completion_tokens"] == 2, "the slots should be free once the streams end"
  http.shutdown()
  hi = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
  assert serve.chat("... <|im_start|> ...", hi) == "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
  assert serve.chat("... <|start_header_id|> ...", hi[1:]) == ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                                                               "<|start_header_id|>assistant<|end_header_id|>\n\n")
  think = serve.server(engine, "smol", "<|im_start|> <think>", port=0)  # a thinking model's template (Qwen3's)
  threading.Thread(target=think.serve_forever, daemon=True).start()
  ask = lambda body: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{think.server_address[1]}/v1/chat/completions",
                                                                   json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
  r = serve.Request(tok, tok.encode(serve.chat("<|im_start|> <think>", hi[1:]), special=True), 6, 0)
  engine.waiting.put(r)
  text, body = tok.decode(done(r)), {"messages": hi[1:], "max_tokens": 6, "temperature": 0}
  message = json.load(ask(body))["choices"][0]["message"]
  assert message == {"role": "assistant", "reasoning_content": text.strip(), "content": ""}  # no </think> came
  events = streamed(ask(body | {"stream": True, "logprobs": True, "top_logprobs": 1}))[0]
  assert "".join(e["delta"].get("reasoning_content", "") for e in events) == text.strip() and not any("content" in e["delta"] for e in events)
  made = [c for e in events if e.get("logprobs") for c in e["logprobs"]["content"]]  # each piece's tokens, with their log probabilities
  joined = b"".join(bytes(c["bytes"]) for c in made).decode("utf-8", "replace")
  assert joined == text and all(c["top_logprobs"][0]["logprob"] == c["logprob"] for c in made)
  # with thinking off, the prompt carries an empty reasoning block, which makes it too long for these slots
  off = len(tok.encode(serve.chat("<|im_start|> <think>", hi[1:], False), special=True))
  with pytest.raises(urllib.error.HTTPError) as e: ask(body | {"chat_template_kwargs": {"enable_thinking": False}})
  assert json.load(e.value)["error"]["message"].startswith(f"{off} tokens")
  think.shutdown()
  assert serve.chat("<|im_start|> <think>", hi[1:], False) == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
  assert serve.answer("<think>\nLet me see.\n</think>\n\nHi!") == ("Let me see.", "Hi!") and serve.answer("<think>\nHm") == ("Hm", "")
  harmony = serve.chat("<|channel|>", hi, effort="low")  # gpt-oss's harmony has its system message, ours as a developer message, and channels
  assert harmony.startswith("<|start|>system<|message|>You are ChatGPT") and "Reasoning: low" in harmony and harmony.endswith(
    "<|start|>developer<|message|># Instructions\n\nBe brief.\n\n<|end|><|start|>user<|message|>Hi<|end|><|start|>assistant")
  assert serve.answer("<|channel|>analysis<|message|>Greet.<|end|><|start|>assistant<|channel|>final<|message|>Hello!") == ("Greet.", "Hello!")
  assert serve.answer("<|channel|>analysis<|message|>Gree") == ("Gree", "") and serve.answer("<|channel|>final<|message|>Hi") == ("", "Hi")
  assert serve.answer("<|channel|>") == serve.answer("<|channel|>analysis") == ("", ""), "a partial channel name is neither reasoning nor answer"

class Kept(serve.Request):  # a request that keeps the logits row behind each of its tokens (Head.row)
  def take(self, head, room):
    self.logits = getattr(self, "logits", []) + [head.row() if isinstance(head, serve.Head) else head]
    return super().take(head, room)

def agrees(r, w, c, tol=0.03):
  # a finished request's logits must be within tol of numpy's, and its picks near numpy's
  ids = r.ids + r.made
  want = llama.forward(ids, list(range(len(ids))), w, c, llama.kv_cache(c, 64, np.zeros), every=True)[len(r.ids) - 1:][:len(r.logits)]
  worst = max(np.abs(g - v).max() / np.abs(v).max() for g, v in zip(r.logits, want))
  assert worst < tol and near(r.logits, want), worst

def streamed(response):  # the choices of a server-sent stream, and the usage from its last chunk
  events = [json.loads(line[6:]) for line in response.read().decode().split("\n\n") if line.startswith("data: {")]
  assert events[-1]["choices"] == [] and "usage" in events[-1], "a stream's last chunk should be its usage"
  return [e["choices"][0] for e in events[:-1]], events[-1]["usage"]

def served(engine, *requests):  # sends requests to an engine, each after the one before ends. The requests in a list go in together
  for batch in requests:
    for r in batch if isinstance(batch, list) else [batch]: engine.waiting.put(r)
    for r in batch if isinstance(batch, list) else [batch]:
      while r.out.get(timeout=600) is not None: pass

@pytest.mark.parametrize("kv", ["bfp8", "bfp4"])
def test_serve_sessions(chip, tmp_path, kv):
  # The serve.py engine with Gated DeltaNet blocks. An agent's next turn resumes from the session kept when its last
  # request ended, or from the one kept after its prompt, and prefills only the new tokens. With bfp4, park 0 leaves
  # no room on the chip, so sessions go to the host and come back in bf16, which holds their values.
  # In bfp4, with 3 bits a value, a key that the chip rounds the other way from numpy moves by an eighth of the largest
  # in its run. numpy's logits move 2.9% when the qkv weights are off by bf16 rounding, against 1.0% in bfp8, so
  # the bound is 6%.
  rt, rng = Runtime(chip), np.random.default_rng(31)
  c, weights = qwen35(rng)
  w, tok, c.precision = llama.arrays(weights), bytes_tokenizer(tmp_path), llama.precision(c, "bf16", kv=kv)
  c.mtp = 0  # decode a token a step. test_serve_speculative drafts with the MTP layer
  engine = serve.Engine(rt, llama.upload(rt, {k: v for k, v in weights.items() if not k.startswith(("mtp.", "layers.4."))}), c, tok,
                        slots=2, places=64, chunk=16, park=0 if kv == "bfp4" else None)
  threading.Thread(target=engine.run, daemon=True).start()
  together = [Kept(tok, p, 6, 0) for p in ([7, 200, 31, 64, 5], [5, 64], list(range(40, 60)))]  # the last is longer than a chunk
  served(engine, together)
  tol = 0.06 if kv == "bfp4" else 0.03
  for r in together: agrees(r, w, c, tol)
  first = together[0]
  for turn, saved in ((first.ids + first.made + [9, 8], first.ids + first.made[:-1]), (first.ids + [1, 2, 3], first.ids)):
    before, r = engine.prefilled, Kept(tok, turn, 4, 0)
    served(engine, r)
    agrees(r, w, c, tol)
    assert engine.prefilled - before == len(turn) - len(saved) == len(turn) - r.cached, (engine.prefilled - before, r.cached)
  assert {type(v[-1]) for v in engine.sessions.values()} == {bytes if kv == "bfp4" else Tensor}, "bfp4 sessions go to the host, bfp8 to the chip"

def test_serve_speculative(chip, tmp_path):
  # The serve.py engine drafts with the multi-token prediction layer of the small Qwen3.5. An oracle's drafts are nearly
  # all accepted, and the random layer's are nearly all rejected. Either way the tokens must agree with numpy, and a
  # next turn resumes from the session with the states from before any rejected draft.
  rt, rng = Runtime(chip), np.random.default_rng(31)
  c, weights = qwen35(rng)
  w, tok, t = llama.arrays(weights), bytes_tokenizer(tmp_path), llama.upload(rt, weights)
  class Oracle(serve.Engine):
    def guess(self, u, best):
      s = self.busy[u].ids + self.busy[u].made
      return int(llama.forward(s, list(range(len(s))), w, c, llama.kv_cache(c, 64, np.zeros))[0].argmax())
  prompts = ([7, 200, 31, 64, 5], [5, 64], list(range(40, 60)))
  for kind in (Oracle, serve.Engine):
    engine = kind(rt, t, c, tok, slots=2, places=64, chunk=16)
    threading.Thread(target=engine.run, daemon=True).start()
    together = [Kept(tok, p, 8, 0) for p in prompts]
    served(engine, together)
    for r in together: agrees(r, w, c)
    assert engine.spec and (engine.agreed >= 0.8 * engine.drafted if kind is Oracle
                            else engine.agreed <= 0.2 * engine.drafted), (engine.agreed, engine.drafted)
  first = together[0]
  before, r = engine.prefilled, Kept(tok, first.ids + first.made + [9, 8], 4, 0)
  served(engine, r)
  agrees(r, w, c)
  assert engine.prefilled - before == 3, engine.prefilled - before

def test_chat_tools(tmp_path):
  # serve.qwen_chat must give the strings that Qwen3.6's chat_template.jinja makes through transformers'
  # apply_chat_template. Tool calls in an answer come out in OpenAI's format, typed by the tool's schema, and go back
  # into the template as they came. Over HTTP, a stand-in engine answers with a tool call, in one response and
  # streamed.
  import queue, urllib.request
  qwen = "<|im_start|> <think> <tool_call>\n<function="  # the markers that identify a template as Qwen3.5's
  tools = [{"type": "function", "function": {"name": "read", "description": "A file's text", "parameters": {"type": "object",
            "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["path"]}}}]
  call = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": {"path": "a.txt", "lines": 2}}}
  messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is in a.txt?"},
              {"role": "assistant", "content": "", "reasoning_content": "I should read it.", "tool_calls": [call]},
              {"role": "tool", "tool_call_id": "call_1", "content": "hello"}]
  want = ("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n" + json.dumps(tools[0]) + "\n</tools>" +
          serve.CALLING + "\n\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is in a.txt?<|im_end|>\n<|im_start|>assistant\n<think>\n"
          "I should read it.\n</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n<parameter=lines>\n2\n</parameter>\n"
          "</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nhello\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")
  assert serve.chat(qwen, messages, tools=tools) == want
  assert serve.chat(qwen, messages[:2]) == ("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is in a.txt?<|im_end|>\n"
                                            "<|im_start|>assistant\n<think>\n")
  assert serve.chat(qwen, messages[:2], False).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "with thinking off, reasoning is empty"
  said = ("I should read it.\n</think>\n\nLet me look.\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n"
          "<parameter=lines>\n2\n</parameter>\n</function>\n</tool_call>")
  text, made = serve.calls(serve.answer(said)[1], tools)
  calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in made]
  assert text == "Let me look." and calls == [("read", {"path": "a.txt", "lines": 2})]
  again = messages[:2] + [{"role": "assistant", "content": text, "reasoning_content": "I should read it.", "tool_calls": made}]
  prompt = serve.chat(qwen, again + messages[3:], tools=tools)
  assert "Let me look.\n\n<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n<parameter=lines>\n2\n</parameter>" in prompt
  class Stand:  # an engine that answers requests with `said`, 7 characters at a time
    tok, places, waiting = bytes_tokenizer(tmp_path), 4096, queue.Queue()
    def offer(self, r): self.waiting.put(r); return True  # there's room for any request
    def run(self):
      while r := self.waiting.get():
        for i in range(0, len(said), 7): r.out.put(said[i:i + 7])
        r.finish = "stop"
        r.out.put(None)
  stand = Stand()
  threading.Thread(target=stand.run, daemon=True).start()
  http = serve.server(stand, "qwen", qwen, port=0)
  threading.Thread(target=http.serve_forever, daemon=True).start()
  ask = lambda body: urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{http.server_address[1]}/v1/chat/completions",
                                                                   json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=60)
  reply = json.load(ask({"messages": messages[:2], "tools": tools}))["choices"][0]
  message = reply["message"]
  assert reply["finish_reason"] == "tool_calls" and message["content"] == "Let me look." and message["reasoning_content"] == "I should read it."
  assert json.loads(reply["message"]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.txt", "lines": 2}
  events = streamed(ask({"messages": messages[:2], "tools": tools, "stream": True}))[0]
  assert "".join(e["delta"].get("content", "") for e in events).strip() == "Let me look." and events[-1]["finish_reason"] == "tool_calls"
  assert [json.loads(c["function"]["arguments"]) for e in events for c in e["delta"].get("tool_calls", [])] == [{"path": "a.txt", "lines": 2}]
  with pytest.raises(urllib.error.HTTPError) as e:
    ask({"messages": messages[:2], "tools": tools, "tool_choice": {"type": "function", "function": {"name": "write"}}})
  assert e.value.code == 400 and "no such tool" in json.load(e.value)["error"]["message"]
  with pytest.raises(urllib.error.HTTPError) as e: ask({"messages": messages[:2], "response_format": {"type": "xml"}})
  assert e.value.code == 400
  http.shutdown()

def test_cost(tmp_path):
  # models/cost.py works from a config alone. Qwen3.6-35B-A3B has 35.5B parameters (0.9B in the multi-token prediction
  # layer) and reads 2.95B a token, about 21.9 GB in the one-card formats. Its KV cache is 20 KB a token (10 attention
  # blocks with 2 KV heads of 256), and its Gated DeltaNet states are 66 MB a session (30 blocks of 32 heads of 128 x
  # 128 in fp32, plus the last 3 conv inputs). Llama-3.1-8B reads benchmark.py's 6.20 GB a token, within 2%. At 256
  # tokens, a prompt chunk reads almost all the experts of each block, so Qwen3.6's 4K-token prompt is over 6 times
  # slower in chunks of 256 than of 2,048. At 32K tokens, attention's reads slow it down again.
  qwen = {"text_config": {"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "num_hidden_layers": 40,
          "rms_norm_eps": 1e-6, "max_position_embeddings": 262144, "vocab_size": 248320, "num_experts": 256, "num_experts_per_tok": 8,
          "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
          "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "mtp_num_hidden_layers": 1,
          "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10, "model_type": "qwen3_5_moe_text",
          "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25, "rope_type": "default"}}}
  llama8b = {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 32, "rms_norm_eps": 1e-5,
             "max_position_embeddings": 131072, "vocab_size": 128256, "intermediate_size": 14336, "rope_theta": 5e5}
  for name, j in (("qwen", qwen), ("llama", llama8b)): (tmp_path / name).mkdir(); (tmp_path / name / "config.json").write_text(json.dumps(j))
  c = llama.config(str(tmp_path / "qwen"))
  s, use = cost.shape(c, "experts"), dict(cost.USE)
  assert 35.4e9 < s.held < 35.6e9 and 2.9e9 < s.values < 3.0e9 and 21.8e9 < s.bytes < 22.2e9, (s.held, s.values, s.bytes)  # includes the draft head
  one = cost.session(c, use | {"context": 1})
  assert (one.kv, one.state) == (10 * 2 * 2 * 256 * 2, 30 * 4 * (32 * 128 * 128 + 3 * (2 * 16 * 128 + 32 * 128)))
  l8 = llama.config(str(tmp_path / "llama"))
  read = cost.shape(l8, "bfp4").every + cost.session(l8, use | {"context": 612.5, "kv": "bfp8"}).kv
  assert abs(read / 6.20e9 - 1) < 0.02, read
  card = dict(cost.CARD)
  t = cost.step(c, s, use, card, 4)
  assert math.isclose(cost.hourly(card) / (4 * t.rate * 3600) * 1e6, cost.hourly(card) / (4 / t.seconds) / 3600 * 1e6)
  (tmp_path / "probe.json").write_text(json.dumps({"DRAM reads of bfp8 chunks, 4 in flight: grid workers, own banks": 420.0,
                                                   "DRAM reads of bfp4 chunks, 4 in flight: grid workers, own banks": 300.0,
                                                   "matmul step, 1 phase, unpacks included": 16.0, "launch, 100 queued: chip each": 2700.0,
                                                   "upload over PCIe": 20.0, "read back over PCIe": 12.0}))
  measured = cost.probed(card, str(tmp_path / "probe.json"))
  assert measured["bandwidth"] == 300e9 and abs(measured["tflops"] / 658e12 - 1) < 0.01 and measured["launch"] == 2e-6 and measured["pcie"] == 12e9
  slow, fast, long = (cost.prompt(c, s, use | {"kv": "bfp8", "context": n}, card, chunk) for n, chunk in ((4096, 256), (4096, 2048), (32768, 2048)))
  assert slow.rate * 6 < fast.rate and long.rate * 1.5 < fast.rate, (slow.rate, fast.rate, long.rate)
  # a chunk of 2,048 reads its activations again for each group of columns. 256 doesn't, since the accumulators fit
  assert cost.regroups(c, 256) == 0 < cost.regroups(c, 2048), cost.regroups(c, 2048)
  assert "a prompt of 8192 tokens" in cost.report(str(tmp_path / "qwen"), "experts", context="8192", kv="bfp8")

def test_fetch(tmp_path, monkeypatch):
  # serve.fetch from a local server standing in for Hugging Face (HF_ENDPOINT). It gets the shards the index names and
  # the configs that exist, and a finished download isn't fetched again.
  import functools, http.server, threading
  hub = tmp_path / "hub" / "org" / "model" / "resolve" / "main"
  hub.mkdir(parents=True)
  weight_map = {"a": "model-1.safetensors", "b": "model-2.safetensors", "c": "model-1.safetensors"}
  files = {"config.json": "{}", "tokenizer.json": "{}", "generation_config.json": "{}", "chat_template.jinja": "<|im_start|>",
           "model-1.safetensors": "1", "model-2.safetensors": "2", "model.safetensors.index.json": json.dumps({"weight_map": weight_map})}
  for name, text in files.items(): (hub / name).write_text(text)
  quiet = type("Quiet", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None})
  hf = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(quiet, directory=str(tmp_path / "hub")))
  threading.Thread(target=hf.serve_forever, daemon=True).start()
  monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{hf.server_address[1]}")
  folder = serve.fetch("org/model", str(tmp_path / "cache"))
  assert sorted(os.listdir(folder)) == sorted([*files, ".done"]) and all((pathlib.Path(folder) / n).read_text() == t for n, t in files.items())
  hf.shutdown()
  assert serve.fetch("org/model", str(tmp_path / "cache")) == folder and serve.fetch(folder) == folder, "fetches once, and uses a folder as is"

@pytest.mark.skipif("LLAMA" not in os.environ, reason="LLAMA=/path/to/SmolLM-135M (Hugging Face HuggingFaceTB/SmolLM-135M) runs it")
def test_llama_model(chip):
  # SmolLM-135M with its pretrained weights, on a prompt and the token decoded after it, against numpy. In fp32, numpy
  # matches Hugging Face to 2e-5. It takes about a minute on ttsim.
  rt = Runtime(chip)
  c, weights = llama.load(os.environ["LLAMA"])
  w, t = llama.arrays(weights), llama.upload(rt, weights)
  kv, np_kv = llama.kv_cache(c, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 32, np.zeros)
  new, places = [504, 2455, 282, 1029, 314], list(range(5))
  for _ in range(2):
    got = llama.logits(llama.forward(Tensor.of(rt, [new], "int32"), Tensor.of(rt, [places], "int32"), t, c, kv))
    want = llama.forward(new, places, w, c, np_kv)[0]
    assert np.abs(got - want).max() < 1 and got.argmax() == want.argmax(), (np.abs(got - want).max(), got.argmax(), want.argmax())
    new, places = [int(want.argmax())], [places[-1] + 1]
  assert new == [1962], "a [question]"
