const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

const OpType = enum {
    add,
    sub,
    mul,
    div,
    pow,
    exp,
};

pub const Value = struct {
    data: f32,
    grad: f32,
    children: [2]?*Value,
    op: ?OpType,
    op_data: f32,

    pub fn init(self: *Value, data: f32) void {
        self.* = .{
            .data = data,
            .grad = 0.0,
            .children = .{ null, null },
            .op = null,
            .op_data = 0.0,
        };
    }

    pub fn add(self: *Value, other: *Value) Value {
        const result = Value{ .data = self.data + other.data, .grad = 0.0, .children = .{ self, other }, .op = OpType.add, .op_data = 0.0 };

        return result;
    }

    pub fn sub(self: *Value, other: *Value) Value {
        const result = Value{ .data = self.data - other.data, .grad = 0.0, .children = .{ self, other }, .op = OpType.sub, .op_data = 0.0 };
        return result;
    }

    pub fn mul(self: *Value, other: *Value) Value {
        const result = Value{ .data = self.data * other.data, .grad = 0.0, .children = .{ self, other }, .op = OpType.mul, .op_data = 0.0 };
        return result;
    }

    pub fn div(self: *Value, other: *Value) Value {
        const result = Value{ .data = self.data / other.data, .grad = 0.0, .children = .{ self, other }, .op = OpType.div, .op_data = 0.0 };
        return result;
    }

    pub fn pow(self: *Value, exponent: f32) Value {
        const result = Value{ .data = math.pow(f32, self.data, exponent), .grad = 0.0, .children = .{ self, null }, .op = OpType.pow, .op_data = exponent };
        return result;
    }

    pub fn exp(self: *Value) Value {
        const result = Value{ .data = @exp(self.data), .grad = 0.0, .children = .{ self, null }, .op = OpType.exp, .op_data = 0.0 };
        return result;
    }

    pub fn add_backward(self: *Value) void {
        assert(self.op == .add);
        assert(self.children[0] != null);
        assert(self.children[1] != null);

        self.children[0].?.grad += self.grad * 1.0;
        self.children[1].?.grad += self.grad * 1.0;
    }

    pub fn mul_backward(self: *Value) void {
        assert(self.op == .mul);
        assert(self.children[0] != null);
        assert(self.children[1] != null);

        self.children[0].?.grad += self.children[1].?.data * self.grad;
        self.children[1].?.grad += self.children[0].?.data * self.grad;
    }

    pub fn pow_backward(self: *Value) void {
        assert(self.op == .pow);
        assert(self.children[0] != null);

        self.children[0].?.grad += self.op_data * math.pow(f32, self.children[0].?.data, self.op_data - 1) * self.grad;
    }

    pub fn exp_backward(self: *Value) void {
        assert(self.op == .exp);
        assert(self.children[0] != null);

        self.children[0].?.grad += self.data * self.grad;
    }

    pub fn sub_backward(self: *Value) void {
        assert(self.op == .sub);
        assert(self.children[0] != null);
        assert(self.children[1] != null);

        self.children[0].?.grad += self.grad;
        self.children[1].?.grad += -self.grad;
    }

    pub fn div_backward(self: *Value) void {
        assert(self.op == .div);
        assert(self.children[0] != null);
        assert(self.children[1] != null);

        const a = self.children[0].?.data;
        const b = self.children[1].?.data;

        self.children[0].?.grad += (1.0 / b) * self.grad;
        self.children[1].?.grad += (-a / (b * b)) * self.grad;
    }

    pub fn build_topo(self: *Value, allocator: std.mem.Allocator) !std.ArrayListUnmanaged(*Value) {
        var visited: std.AutoHashMapUnmanaged(*Value, void) = .{};
        defer visited.deinit(allocator);

        var topo: std.ArrayListUnmanaged(*Value) = .empty;
        try self.build_topo_recursive(&visited, &topo, allocator);
        std.mem.reverse(*Value, topo.items);

        return topo;
    }

    fn build_topo_recursive(self: *Value, visited: *std.AutoHashMapUnmanaged(*Value, void), topo: *std.ArrayListUnmanaged(*Value), allocator: std.mem.Allocator) !void {
        if (visited.contains(self)) return;

        try visited.put(allocator, self, {});

        for (self.children) |maybe_child| {
            if (maybe_child) |child| {
                try child.build_topo_recursive(visited, topo, allocator);
            }
        }

        try topo.append(allocator, self);
    }

    pub fn backward(self: *Value, allocator: std.mem.Allocator) !void {
        self.grad = 1.0;
        var topo = try build_topo(self, allocator);
        defer topo.deinit(allocator);

        for (topo.items) |node| {
            if (node.op) |operation| {
                switch (operation) {
                    .add => node.add_backward(),
                    .sub => node.sub_backward(),
                    .mul => node.mul_backward(),
                    .div => node.div_backward(),
                    .pow => node.pow_backward(),
                    .exp => node.exp_backward(),
                }
            }
        }
    }

    // very shaky impl, maybe the future me will regret this
    pub fn debug(self: Value) void {
        std.debug.print("Value(data={d:.4}", .{self.data});
        var has_children = false;
        for (self.children) |maybe_child| {
            if (maybe_child != null) {
                if (!has_children) {
                    std.debug.print(", children=[", .{});
                    has_children = true;
                } else {
                    std.debug.print(", ", .{});
                }
                std.debug.print("{d:.4}", .{maybe_child.?.data});
            }
        }
        if (has_children) std.debug.print("]", .{});

        std.debug.print(", op={any})\n", .{self.op});
    }
};
