const std = @import("std");
const engine = @import("engine.zig");

pub const Neuron = struct {
    weight: std.ArrayListUnmanaged(*engine.Value),
    bias: engine.Value,
};
