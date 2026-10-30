# First P150a workload

The first release target is **one P150a running Llama-3.1-8B-Instruct for one
user**, with greedy text completion. SmolLM-135M is the smaller development
fixture. Passing its tests does not qualify Llama or the card.

This is the acceptance specification for [issue #3](https://github.com/baretile/baretile/issues/3),
dated October 29, 2026. The thresholds below are requirements, not measured
results. Hardware correctness and performance remain unverified.

## Fixed inputs

Use these exact revisions, including their tokenizers. [workload.sha256](workload.sha256)
holds the checksum of every file.

| Input | Revision |
| --- | --- |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) | `0e9e39f249a16976918f6564b8830bc894c89659` |
| [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M/tree/1d461723eec654e65efdc40cf49301c89c0c92f4) | `1d461723eec654e65efdc40cf49301c89c0c92f4` |
| [Tenstorrent reference tokens](https://github.com/tenstorrent/tt-metal/blob/9f98dee9c5a4227b3fd866aa6662e02ba2b362af/models/tt_transformers/tests/reference_outputs/Llama-3.1-8B-Instruct.refpt) and book text | tt-metal `9f98dee9c5a4227b3fd866aa6662e02ba2b362af` |

The reference contains 1,024 book tokens and their next-token predictions.
Use its first 512 tokens verbatim as the prompt, including its initial BOS
token. Score predictions starting at prompt position 511. Do not retokenize
this prompt or add a chat template. The file fixes the expected predictions;
its upstream generation environment is not fully recorded, so it does not
replace an independent model check.

Run the commands from a baretile checkout. Keep model files outside it. Install
`huggingface_hub` for `hf download`; NumPy and pytest run the smoke checks,
and PyTorch reads the `.refpt` file. Llama downloads require access to the
gated model through Hugging Face. Use an existing authorized `hf` login.

```sh
export WORKLOAD_DIR=/absolute/path/to/workload-files
export LLAMA_DIR="$WORKLOAD_DIR/Llama-3.1-8B-Instruct"
export SMOLLM_DIR="$WORKLOAD_DIR/SmolLM-135M"
export REF_DIR="$WORKLOAD_DIR/reference"
mkdir -p "$REF_DIR"

hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --local-dir "$LLAMA_DIR" \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json model.safetensors.index.json \
  model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors
hf download HuggingFaceTB/SmolLM-135M \
  --revision 1d461723eec654e65efdc40cf49301c89c0c92f4 \
  --local-dir "$SMOLLM_DIR" \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json model.safetensors

TT_REFERENCE_URL=https://raw.githubusercontent.com/tenstorrent/tt-metal/9f98dee9c5a4227b3fd866aa6662e02ba2b362af/models/tt_transformers/tests
curl --fail --location --max-time 120 \
  "$TT_REFERENCE_URL/reference_outputs/Llama-3.1-8B-Instruct.refpt" \
  --output "$REF_DIR/Llama-3.1-8B-Instruct.refpt"
curl --fail --location --max-time 120 \
  "$TT_REFERENCE_URL/tale-of-two-cities.txt.bz2" \
  --output "$REF_DIR/tale-of-two-cities.txt.bz2"

(cd "$WORKLOAD_DIR" && sha256sum --check) < docs/workload.sha256
```

Stop if any checksum fails. Use clean model directories: an extra
`chat_template.jinja` or edited generation config changes the input. Pass these
local folders to the model commands; the server's model-ID download path does
not pin a revision.

## Precision and limits

All layers and the full vocabulary participate: 32 layers for Llama and 30 for
SmolLM. No layer cutting, vocabulary reduction or CPU fallback qualifies.
The model's CPU tokenizer and sampling are allowed; transformer computations
run on the card. The comparison uses baretile's `tt` mode, defined in
[`llama.precision`](../baretile/models/llama.py), against a separate `bf16` diagnostic.

| Operation | Llama `tt` | `bf16` diagnostic / SmolLM fixture |
| --- | --- | --- |
| Q/K/V, attention output, MLP down, output head | bfp8 weights, 2 fidelity phases | bf16 weights, 4 phases |
| MLP gate/up, layers 0–30 | bfp4 weights, 1 phase | bf16 weights, 4 phases |
| MLP gate/up, layer 31 | bfp8 weights, 2 phases | bf16 weights, 4 phases |
| Attention products | 4 phases in prefill, 2 in decode | 4 phases |
| KV cache | bfp8 | bf16 |
| Embeddings, norms, RoPE tables | bf16 | bf16 |

Token and position tensors are int32. The residual stream between blocks is
fp32: attention's output and the MLP's down keep their fp32 sums and add them
to it, and each norm reads it rounded to bf16. Other stored activations are
bf16; matmul accumulators and the final logits are fp32. Fused intermediates need not
round at the same boundaries as separate operations; [compiler](compiler.md)
describes these limits. Record the baretile commit as part of the precision
definition. NumPy's same-weight reference uses full products and does not
reproduce every intermediate rounding.

Batch size is 1. The initial serving limit is 512 prompt tokens and 32 output
tokens, in one slot of 1,024 positions. B2 accuracy needs a 1,024-row cache;
B2 decode allocates 736 rows for its shorter run. Longer contexts, batch 32,
multi-card runs, training, adapters, chat/tool templates, sampling at nonzero
temperature and other models are outside this qualification.

## Correctness gates

Every gate must pass on the relevant target. A skip, timeout, nonfinite logit,
missing comparison or incomplete run is not a pass. Keep failures in the
record. Change a threshold only through a reviewed specification change with
the failing evidence and a numerical explanation, before resuming optimization.

| Gate | Required result |
| --- | --- |
| Tokenizer | Exact token IDs and decoded text against Hugging Face's tokenizer from the same revision, with `add_special_tokens=False`, for the three prompts below and `"  café İstanbul 12345\n\tHello!"` |
| SmolLM checkpoint smoke | Existing `test_llama_model` passes: two decisions match NumPy's argmax, maximum absolute logit error is below 1, and the second chosen token is 1962 |
| SmolLM autonomous generation, bf16 | Three prompts × 32 decisions; each run feeds its own selected tokens back. At every step, relative maximum logit error is at most 3%, and the selected token is within 1% of the reference logit span of the reference maximum |
| Llama B2 reference agreement, `tt` | At least 450/500 top-1 matches (90%) and 490/500 top-5 matches (98%), using integer counts without rounding percentages upward |
| Llama autonomous generation, `tt` | Three prompts × 32 decisions, using the run's own history; its chosen token must be in the independent bf16 reference's top five at every step |
| Single-slot HTTP | All three prompts complete both with and without streaming, within the limits below, with identical greedy output for the two response forms |

The B2 floors match the pinned vendor's
[P150 targets](https://github.com/tenstorrent/tt-metal/blob/9f98dee9c5a4227b3fd866aa6662e02ba2b362af/models/model_targets.yaml#L685).
They are adopted requirements, not a claim of parity with a measured vendor
run. The autonomous gates are additional baretile requirements.

Use these literal UTF-8 prompts, without BOS, a chat template or trailing
newline, for autonomous and HTTP checks:

```text
The meaning of life is
The capital of France is
1, 2, 3,
```

For the 32-decision diagnostic, disable EOS stopping and other stop rules.
For HTTP, honor the pinned generation config's EOS tokens and stop at EOS or
32 output tokens. Set temperature 0, top_k 0, top_p 1, min_p 0, repetition
penalty 1, and presence/frequency penalties 0. No speculative decoding.

For SmolLM, score each logit row against an independent Transformers CPU
reference, loaded from the pinned checkpoint, rounded to bf16 weights and then
cast to fp32 for eager arithmetic. Use the exact prefix produced by baretile
at that step; never replace its next input with the reference's choice.
Relative maximum error is `max(abs(got - ref)) / max(max(abs(ref)), 1e-6)`.
Selection loss is `(max(ref) - ref[pick]) / max(max(ref) - min(ref), 1e-6)`.
Also compare the NumPy path with that reference at the same prefixes, to
separate loader/model errors from device errors. For Llama's independent bf16
reference, use Transformers in bf16 with eager attention and record its device
and package versions. Do not substitute baretile's own model code for this check.

Keep the Llama bf16 and same-quantized-weight NumPy B2 results beside the `tt`
result. The former exposes loss introduced by the low-precision configuration;
the latter helps distinguish weight conversion from device arithmetic. Neither
comparison alone proves that the device implementation is correct.

## Commands and remaining checks

First run the small checkpoint smoke on ttsim. This exercises the full SmolLM
checkpoint but only two decisions. It does not complete the autonomous gate.

```sh
export TTSIM=/absolute/path/to/libttsim_bh.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LLAMA="$SMOLLM_DIR" \
  python3 -m pytest -s -ra test/test_models.py::test_llama_model \
  test/test_models.py::test_llama_precision \
  test/test_models.py::test_token_accuracy
```

The ordinary `python3 -m baretile.models.llama model "prompt"` demonstration
follows NumPy's choices, so it isn't the autonomous test. `test/generation.py` is:
it feeds the chip its own picks for the three prompts, compares every step with
transformers at the same prefix, and exits with an error when a step misses its
gate. In `bf16` that's the SmolLM gate above, and in `tt` the Llama one. It needs
PyTorch and transformers, and writes each step's errors to the report:

```sh
python3 test/generation.py "$SMOLLM_DIR" smollm-generation.json 32 bf16
python3 test/generation.py "$LLAMA_DIR" llama-generation.json 32 tt
```

Running the independent reference checks in CI is tracked in
[#5](https://github.com/baretile/baretile/issues/5).

On a card, complete [bring-up](bring-up.md) first and run with `TTSIM` unset
in the root environment described there. These existing commands report B2
accuracy; compare their reported percentages with the gate, since the CLI does
not enforce the threshold or return a failure code for a low score.

```sh
python3 -m baretile.models.llama "$LLAMA_DIR" "$REF_DIR/Llama-3.1-8B-Instruct.refpt" 500 tt
python3 -m baretile.models.llama "$LLAMA_DIR" "$REF_DIR/Llama-3.1-8B-Instruct.refpt" 500 bf16
python3 -m baretile.models.llama "$LLAMA_DIR" "$REF_DIR/Llama-3.1-8B-Instruct.refpt" 500 tt numpy
```

For the HTTP case, start one slot, 1,024 positions, `tt` precision and bfp8 KV:

```sh
python3 -m baretile.models.serve "$LLAMA_DIR" 8000 1 1024 tt bfp8
```

In another terminal, submit each of the three prompts, one request at a time:

```sh
curl --fail-with-body --max-time 600 http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The meaning of life is","max_tokens":32,"temperature":0,"top_k":0,"top_p":1,"min_p":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"stream":false}'
```

Repeat with `"stream":true,"stream_options":{"include_usage":true}`. Require
HTTP 200, nonempty output, a normal stop/length finish reason, correct usage
counts, and equal assembled text and token counts across the two response
forms. A failed or cancelled request must release the slot so another request
can finish. The HTTP checks, including all three prompts and slot recovery,
belong in the [card acceptance record](https://github.com/baretile/baretile/issues/11).
The server binds to all interfaces and has no API-key check; run this local
acceptance case on a trusted host as described in [serving](models.md#serving).

## Timing and evidence

Preserve the existing B2 protocol: prefill 512 reference tokens, then run 200
greedy decode steps, feeding each selected token back and ignoring EOS. Steps
0 and 1 execute/record the JIT and are excluded from the rate. The numerator
is **198 timed steps**, not 200. Each interval begins with token IDs on the
host and ends after reading logits back and selecting the next token on the
host. Loading, upload and prompt prefill are outside that interval. Report
model load/upload time and prompt-to-first-token latency separately.

Run three fresh processes on the card and retain each raw result plus the
median. Record cold versus reused compilation cache. Record PCIe link, firmware,
available workers, host, Python/NumPy/PyTorch/Transformers versions, model hashes,
baretile and vendor commits, precision and command. HTTP first-text latency is
time to the first nonempty SSE event, not a per-token timestamp.

The existing broad benchmark command is:

```sh
BOOK="$REF_DIR/tale-of-two-cities.txt.bz2" \
  python3 test/benchmark.py baretile "$LLAMA_DIR" "$REF_DIR/Llama-3.1-8B-Instruct.refpt"
```

It also runs B1, batch-32 B2, long-context B3 and multi-user B4. Those results
are outside this specification; it is not yet an isolated first-workload
command. [Benchmark task #12](https://github.com/baretile/baretile/issues/12)
must provide that isolation and a comparable vendor measurement before a
performance claim. `--quick` cuts layers and sizes, so it cannot qualify a model.
There is no tokens/s release floor yet: establish the hardware baseline before
choosing an optimization target. Simulator times do not estimate card speed.

Publish a result for each gate as passed, failed or not run, with commands,
logs and the exact source revision. This specification can be complete while
the release remains unqualified. Changing an input, threshold or timing
boundary creates a new comparison; preserve the earlier record.
