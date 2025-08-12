const std = @import("std");
const eng = @import("engine.zig");
const nn = @import("nn.zig");

pub fn main() !void {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var tape = eng.Tape(20000){};

    // model: MLP(3, [4,4,1])
    var model: nn.MLP(3, 4, 4, 1, 20000) = undefined;
    model.init(&tape, rng);

    const param_mark = tape.len;

    const xs = [_][3]f32{
        .{ 2.0, 3.0, -1.0 },
        .{ 3.0, -1.0, 0.5 },
        .{ 0.5, 1.0, 1.0 },
        .{ 1.0, 1.0, -1.0 },
    };
    const ys = [_]f32{ 1.0, -1.0, -1.0, 1.0 };

    var k: usize = 0;
    while (k < 20) : (k += 1) {
        tape.len = param_mark;

        var loss: *eng.Value = undefined;
        var first = true;

        var i: usize = 0;
        while (i < xs.len) : (i += 1) {
            const x0 = tape.new_value(xs[i][0]);
            const x1 = tape.new_value(xs[i][1]);
            const x2 = tape.new_value(xs[i][2]);

            const out_arr = model.forward(.{ x0, x1, x2 });
            const ypred = out_arr[0];

            const ygt = tape.new_value(ys[i]);
            const neg1 = tape.new_value(-1.0);
            const diff = tape.add(ypred, tape.mul(ygt, neg1));
            const sq = tape.mul(diff, diff);

            if (first) {
                loss = sq;
                first = false;
            } else {
                loss = tape.add(loss, sq);
            }
        }

        tape.backward(loss);

        model.for_each_param(struct {
            fn each(p: *eng.Value) void {
                p.data -= 0.1 * p.grad;
            }
        }.each);

        std.debug.print("{d} loss={d:.6}\n", .{ k, loss.data });
    }
}
