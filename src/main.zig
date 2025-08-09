const std = @import("std");
const engine = @import("engine.zig");

pub fn main() !void {
    var buffer: [1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buffer);
    const allocator = fba.allocator();

    // inputs x1, x2
    var x1: engine.Value = undefined;
    x1.init(2.0);
    var x2: engine.Value = undefined;
    x2.init(0.0);

    // weights w1, w2
    var w1: engine.Value = undefined;
    w1.init(-3.0);
    var w2: engine.Value = undefined;
    w2.init(1.0);

    // bias
    var b: engine.Value = undefined;
    b.init(6.8813735870195432);

    // x1*w1 + x2*w2 + b
    var x1w1 = x1.mul(&w1);
    var x2w2 = x2.mul(&w2);
    var x1w1x2w2 = x1w1.add(&x2w2);
    var n = x1w1x2w2.add(&b);

    var o = n.tanh();
    try o.backward(allocator);

    std.debug.print("Result: {d:.6}\n", .{o.data});
    std.debug.print("x1.grad: {d:.6}\n", .{x1.grad});
    std.debug.print("x2.grad: {d:.6}\n", .{x2.grad});
    std.debug.print("w1.grad: {d:.6}\n", .{w1.grad});
    std.debug.print("w2.grad: {d:.6}\n", .{w2.grad});
    std.debug.print("b.grad: {d:.6}\n", .{b.grad});
}
