// main.zig
const std = @import("std");
const engine = @import("engine.zig");

pub fn main() !void {
    var tape = engine.Tape(128){};
    // inputs x1, x2
    const x1 = tape.new_value(2.0);
    const x2 = tape.new_value(0.0);

    // weights w1, w2
    const w1 = tape.new_value(-3.0);
    const w2 = tape.new_value(1.0);

    // bias
    const b = tape.new_value(6.8813735870195432);

    // x1*w1 + x2*w2 + b
    const x1w1 = tape.mul(x1, w1);
    const x2w2 = tape.mul(x2, w2);
    const sum = tape.add(x1w1, x2w2);
    const n = tape.add(sum, b);
    const o = tape.tanh(n);

    tape.backward(o);

    std.debug.print("Result:   {d:.6}\n", .{o.data});
    std.debug.print("x1.grad:  {d:.6}\n", .{x1.grad});
    std.debug.print("x2.grad:  {d:.6}\n", .{x2.grad});
    std.debug.print("w1.grad:  {d:.6}\n", .{w1.grad});
    std.debug.print("w2.grad:  {d:.6}\n", .{w2.grad});
    std.debug.print("b.grad:   {d:.6}\n", .{b.grad});
}
