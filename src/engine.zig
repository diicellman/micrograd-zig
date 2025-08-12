const std = @import("std");
const assert = std.debug.assert;

pub const INVALID: u32 = std.math.maxInt(u32);

pub const OpType = enum(u8) {
    none,
    add,
    mul,
    tanh,
};

pub const Value = struct {
    data: f32,
    grad: f32,
    op_data: f32,
    child0: u32,
    child1: u32,
    id: u32,
    op: OpType,
};

pub fn Tape(comptime max_nodes: u32) type {
    return struct {
        const Self = @This();

        nodes: [max_nodes]Value = undefined,
        len: u32 = 0,

        pub fn reset(self: *Self) void {
            self.len = 0;
        }

        pub fn new_value(self: *Self, data: f32) *Value {
            assert(self.len < max_nodes);
            const i = self.len;
            self.len = i + 1;
            self.nodes[i] = .{
                .data = data,
                .grad = 0.0,
                .op_data = 0.0,
                .child0 = INVALID,
                .child1 = INVALID,
                .id = i,
                .op = .none,
            };
            return &self.nodes[i];
        }

        pub fn add(self: *Self, a: *Value, b: *Value) *Value {
            const out = self.new_value(a.data + b.data);
            out.child0 = a.id;
            out.child1 = b.id;
            out.op = .add;
            return out;
        }

        pub fn mul(self: *Self, a: *Value, b: *Value) *Value {
            const out = self.new_value(a.data * b.data);
            out.child0 = a.id;
            out.child1 = b.id;
            out.op = .mul;
            return out;
        }

        pub fn tanh(self: *Self, x: *Value) *Value {
            const out = self.new_value(std.math.tanh(x.data));
            out.child0 = x.id;
            out.child1 = INVALID;
            out.op = .tanh;
            return out;
        }

        pub fn zero_grads(self: *Self) void {
            var i: u32 = 0;
            while (i < self.len) : (i += 1) self.nodes[i].grad = 0.0;
        }

        pub fn backward(self: *Self, root: *Value) void {
            var visited: [max_nodes]bool = [_]bool{false} ** max_nodes;
            var s1: [max_nodes]*Value = undefined;
            var sp1: u32 = 0;
            var s2: [max_nodes]*Value = undefined;
            var sp2: u32 = 0;

            // DFS
            s1[sp1] = root;
            sp1 += 1;
            while (sp1 != 0) {
                sp1 -= 1;
                const v = s1[sp1];
                if (visited[v.id]) continue;
                visited[v.id] = true;

                s2[sp2] = v;
                sp2 += 1;

                if (v.child0 != INVALID) {
                    const c0 = &self.nodes[v.child0];
                    s1[sp1] = c0;
                    sp1 += 1;
                }
                if (v.child1 != INVALID) {
                    const c1 = &self.nodes[v.child1];
                    s1[sp1] = c1;
                    sp1 += 1;
                }
            }

            self.zero_grads();
            root.grad = 1.0;

            var i: u32 = 0;
            while (i < sp2) : (i += 1) {
                const v = s2[i];
                switch (v.op) {
                    .none => {},
                    .add => {
                        const a = &self.nodes[v.child0];
                        const b = &self.nodes[v.child1];
                        a.grad += v.grad;
                        b.grad += v.grad;
                    },
                    .mul => {
                        const a = &self.nodes[v.child0];
                        const b = &self.nodes[v.child1];
                        a.grad += b.data * v.grad;
                        b.grad += a.data * v.grad;
                    },
                    .tanh => {
                        const x = &self.nodes[v.child0];
                        const d = 1.0 - (v.data * v.data);
                        x.grad += d * v.grad;
                    },
                }
            }
        }
    };
}
