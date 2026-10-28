# Use tensors on the chip: operations, fusion, shapes, formats, views, weight storage, gradients and jit. Ramp 2 goes
# below the tensors to the kernels that run them.
#   TTSIM=/path/to/libttsim.so python3 ramps/1_tensors.py
# See README.md for setup and docs/README.md for the other ramps and the documentation.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baretile import pci, tensor
from baretile.chip import Chip
from baretile.runtime import Runtime
from baretile.tensor import Tensor, jit

# 1. Open the device and create a runtime. A Tensix tile contains RISC-V cores and local memory.
# The last tile dispatches work; the others run the kernels as workers.
chip = Chip(pci.open_device())
rt = Runtime(chip)
print(f"1. {len(rt.cores)} workers, dispatcher at {rt.dispatcher}")

# 2. Tensor operations build an expression. tolist() compiles it, runs the kernels and reads the result.
# The matmul and ReLU below share a kernel. rt.profile(since) groups launches by name, starting at since.
x = Tensor.of(rt, [[1.0, 2.0], [3.0, 4.0]])
y = (x @ x).relu()
assert y.op == "relu" and y.srcs[0].op == "matmul"  # the matmul and ReLU haven't run yet
since = rt.launches
assert y.tolist() == [[7.0, 10.0], [15.0, 22.0]]
assert sorted(name for name, *_ in rt.profile(since)) == ["matmul relu", "readback"]
print("2. (x @ x).relu() ran as one kernel")

# 3. Softmax combines subtraction, exponentiation and reductions in one kernel. realize() queues it without
# reading the result. The expression is part of its key in tensor.kernels; 0 refers to the input x.
# To see the expression as a graph, run python3 -m baretile.viz ramps/1_tensors.py and open viz.html.
since = rt.launches
x.softmax().realize()
e = list(tensor.kernels)[-1][0]  # expression from the last kernel added to the cache
assert rt.launches == since + 1 and e == ("mul", ("exp", ("sub", 0, ("max2", 0))), ("recip", ("sum2", ("exp", ("sub", 0, ("max2", 0))))))
# Operations that use a reduction's result need a later pass, which can repeat a fused matmul. Before reductions,
# we compute matmuls separately when K exceeds 256. K is the dimension summed over. "T" means B is read transposed.
for k, want in ((256, ["T matmul sum2"]), (512, ["T matmul", "sum2"])):
  a = Tensor.of(rt, [[1.0] * k for _ in range(64)])
  since = rt.launches
  (a @ a.T).sum(2).realize()
  assert sorted(name for name, *_ in rt.profile(since)) == want
print(f"3. softmax kernel expression: {e}")

# 4. Shapes are (batch, rows, cols). sum(0) adds across the batch, sum(1) adds down the rows, and sum(2) adds
# across each row. Broadcasting follows NumPy: the single row below is added to both rows of x.
pair = Tensor.of(rt, [[[1.0, 2.0]], [[3.0, 4.0]]])  # a batch of two 1 x 2 matrices
assert pair.shape == (2, 1, 2) and pair.sum(0).tolist() == [[4.0, 6.0]]
assert x.sum(1).tolist() == [[4.0, 6.0]] and x.sum(2).tolist() == [[3.0], [7.0]]
assert (x + Tensor.of(rt, [[10.0, 20.0]])).tolist() == [[11.0, 22.0], [13.0, 24.0]]
print("4. summed each axis and added one row to both rows of x")

# 5. Tensors use bf16 by default, with 8 bits of precision. Here 1/3 rounds to 0.333984375. Use fp32 for 24 bits
# of precision, or int32 for indices and integer arithmetic. int32 addition returns int32. Operations without
# an integer implementation, such as multiplication, convert to fp32.
assert Tensor.of(rt, [[1 / 3]]).tolist() == [[0.333984375]]
assert abs(Tensor.of(rt, [[1 / 3]], "fp32").tolist()[0][0] - 1 / 3) < 1e-7
ids = Tensor.of(rt, [[7, -2]], "int32") + 5
assert ids.dtype == "int32" and ids.tolist() == [[12, 3]]
print("5. bf16, fp32 and int32")

# 6. Tensors are stored in 32 x 32 tiles of values. A view shares the original tensor's memory. Selecting a batch
# entry or slicing whole tiles from a stored tensor needs no kernel. big[1:3, :] cuts through a tile, so a move
# kernel copies those rows.
big = Tensor.of(rt, [[float(r % 8)] * 64 for r in range(64)])
since = rt.launches
top, second = big[:32, :], pair[1]
assert rt.launches == since and top.owner is big and top.shape == (1, 32, 64) and second.owner is pair
rows = big[1:3, :]
assert rows.op == "move" and rows.tolist() == [[1.0] * 64, [2.0] * 64]
print("6. top and second share their inputs' memory; rows was copied")

# 7. Matmul can read B in bfp8 or bfp4. These formats share one exponent among 16 values, with 7 magnitude bits
# per value in bfp8 and 3 in bfp4. columns() stores each column of tiles together in a DRAM bank so workers can
# read several tiles per request. The models use this layout for weights.
# The weights below fit exactly in bfp8 and the sums fit in bf16, so we can compare the results with Python using ==.
am = [[float((r + c) % 2) for c in range(64)] for r in range(32)]
bm = [[float(r * c % 3) for c in range(32)] for r in range(64)]
want = [[sum(am[r][k] * bm[k][c] for k in range(64)) for c in range(32)] for r in range(32)]
A, B = Tensor.of(rt, am), Tensor.of(rt, bm)
B8 = B.cast("bfp8").realize()
assert (A @ B).tolist() == want and (A @ B8).tolist() == want and (A @ B8.columns()).tolist() == want
print("7. bf16, bfp8 and bfp8 stored by columns gave the same matmul result")

# 8. requires_grad_() enables gradients for v. backward() accumulates them in v.grad.
# backward() queues the gradients before returning; tolist() waits and reads them.
v = Tensor.of(rt, [[1.0, 2.0, 3.0]], "fp32").requires_grad_()
(v * v).sum().backward()
assert v.grad.tolist() == [[2.0, 4.0, 6.0]]
print("8. the gradient of sum(v * v) is 2v")

# 9. Fit y = 3x + 1 with gradient descent. assign() updates w and b in place after each step.
# jit runs step normally once, records the second call's launches, then replays them with one doorbell write.
# On replay, step's Python code doesn't run. Input shapes and dtypes must stay fixed.
xs = Tensor.of(rt, [[i / 7 for i in range(8)]], "fp32")
ys = xs * 3 + 1
w, b = (Tensor.of(rt, [[0.0]], "fp32").requires_grad_() for _ in range(2))
@jit
def step(xs, ys):
  w.grad = b.grad = None
  d = xs * w + b - ys
  loss = (d * d).mean()
  loss.backward()
  w.assign(w - w.grad * 0.5)
  b.assign(b - b.grad * 0.5)
  return loss
for _ in range(80): loss = step(xs, ys)
since = rt.launches
step(xs, ys)
assert abs(w.tolist()[0][0] - 3) < 0.01 and abs(b.tolist()[0][0] - 1) < 0.01 and loss.tolist()[0][0] < 1e-4
print(f"9. after 81 steps: w = {w.tolist()[0][0]:.4f}, b = {b.tolist()[0][0]:.4f}; {rt.launches - since} launches for the last step and readbacks")

# See ramps/2_runtime.py for the kernels under these tensors, and models/gpt2.py for a model built with Tensor.
print(f"the ramp ran {rt.launches} launches")
