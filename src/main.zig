const std = @import("std");
const engine = @import("engine.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // inputs x1, x2
    var x1 = engine.Value{ .data = 2.0, .grad = 0.0, .children = .{ null, null }, .op = null, .op_data = 0.0 };
    var x2 = engine.Value{ .data = 0.0, .grad = 0.0, .children = .{ null, null }, .op = null, .op_data = 0.0 };

    // weights w1, w2
    var w1 = engine.Value{ .data = -3.0, .grad = 0.0, .children = .{ null, null }, .op = null, .op_data = 0.0 };
    var w2 = engine.Value{ .data = 1.0, .grad = 0.0, .children = .{ null, null }, .op = null, .op_data = 0.0 };

    // bias
    var b = engine.Value{ .data = 6.8813735870195432, .grad = 0.0, .children = .{ null, null }, .op = null, .op_data = 0.0 };

    // x1*w1 + x2*w2 + b
    var x1w1 = x1.mul(&w1);
    var x2w2 = x2.mul(&w2);
    var x1w1x2w2 = x1w1.add(&x2w2);
    var n = x1w1x2w2.add(&b);

    // e = (2*n).exp()
    var two = engine.Value{ .data = 2.0, .grad = 0.0, .children = .{ null, null }, .op = null, .op_data = 0.0 };
    var two_n = two.mul(&n);
    var e = two_n.exp();

    // o = (e - 1) / (e + 1)
    var one = engine.Value{ .data = 1.0, .grad = 0.0, .children = .{ null, null }, .op = null, .op_data = 0.0 };
    var one2 = engine.Value{ .data = 1.0, .grad = 0.0, .children = .{ null, null }, .op = null, .op_data = 0.0 };
    var e_minus_1 = e.sub(&one);
    var e_plus_1 = e.add(&one2);
    var o = e_minus_1.div(&e_plus_1);

    try o.backward(allocator);

    std.debug.print("Result: {d:.6}\n", .{o.data});
    std.debug.print("x1.grad: {d:.6}\n", .{x1.grad});
    std.debug.print("x2.grad: {d:.6}\n", .{x2.grad});
    std.debug.print("w1.grad: {d:.6}\n", .{w1.grad});
    std.debug.print("w2.grad: {d:.6}\n", .{w2.grad});
    std.debug.print("b.grad: {d:.6}\n", .{b.grad});
}
