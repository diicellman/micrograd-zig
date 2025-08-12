# micrograd-zig

_[LOGIC — Medium: Success]_ You piece it together: a tiny autodiff, big lessons.

Karpathy’s micrograd – but in Zig. This is a small, educational rewrite with a tape based engine and a toy MLP on top. It’s meant for learning purposes!

## Repo structure

- **engine.zig** – a minimal reverse-mode autodiff "tape" with fixed capacity (no runtime allocs).
- **nn.zig** – a tiny MLP built on the engine (`add`, `mul`, `tanh` only for now).
- **main.zig** – an example training loop inspired by Karpathy’s video.

## Quick start

Clone and run the example:

```bash
zig run src/main.zig
```

You should see the loss decreasing over iterations.

## Example (from `main.zig`)

This mirrors the small dataset from the video:

```zig
// dataset
const xs = [_][3]f32{
    .{ 2.0,  3.0, -1.0 },
    .{ 3.0, -1.0,  0.5 },
    .{ 0.5,  1.0,  1.0 },
    .{ 1.0,  1.0, -1.0 },
};
const ys = [_]f32{ 1.0, -1.0, -1.0, 1.0 };

// model: MLP(3, [4, 4, 1]) with tanh activations
var model: nn.MLP(3, 4, 4, 1, 100_000) = undefined;
model.init(&tape, rng);

// training loop (sum of squared errors + SGD)
const param_mark = tape.len;
var step: usize = 0;
while (step < 20) : (step += 1) {
    tape.len = param_mark;

    // forward
    var loss: *engine.Value = undefined;
    var first = true;
    inline for (xs, 0..) |x, i| {
        const x0 = tape.new_value(x[0]);
        const x1 = tape.new_value(x[1]);
        const x2 = tape.new_value(x[2]);

        const out = model.forward(.{ x0, x1, x2 })[0];
        const ygt = tape.new_value(ys[i]);
        const diff = tape.add(out, tape.mul(ygt, tape.new_value(-1.0)));
        const sq   = tape.mul(diff, diff);

        loss = if (first) blk: { first = false; break :blk sq; }
               else            tape.add(loss, sq);
    }

    // backward + update
    tape.backward(loss);
    model.for_each_param(struct {
        fn each(p: *engine.Value) void { p.data -= 0.1 * p.grad; }
    }.each);

    std.debug.print("{d} loss={d:.6}", .{ step, loss.data });
}
```

## Thanks

This project exists purely to understand how autodiff works. Credit to Andrej Karpathy for the original [micrograd](https://github.com/karpathy/micrograd). Any mistakes are product of my degeneracy.
