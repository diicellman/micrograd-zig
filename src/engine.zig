const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

const OpType = enum {
    add,
    // sub,
    mul,
    // div,
    // pow,
    // exp,
    tanh,
};

pub const Value = struct {
    data: f32,
    grad: f32,
    children: [2]?*Value,
    op: ?OpType,
    op_data: f32,
    id: u32,
};

pub fn Tape(comptime max_nodes: u32) type {
    return struct {
        const Self = @This();

        nodes: [max_nodes]Value = undefined,
        len: u32 = 0,

        pub fn new_value(self: *Self, data: f32) *Value {
            assert(self.len < max_nodes);
            const i = self.len;
            self.len += 1;
            self.nodes[i] = .{
                .data = data,
                .grad = 0.0,
                .children = .{ null, null },
                .op = null,
                .op_data = 0.0,
                .id = i,
            };
            return &self.nodes[i];
        }

        pub fn add(self: *Self, a: *Value, b: *Value) *Value {
            const out = self.new_value(a.data + b.data);
            out.children = .{ a, b };
            out.op = .add;
            return out;
        }

        pub fn mul(self: *Self, a: *Value, b: *Value) *Value {
            const out = self.new_value(a.data * b.data);
            out.children = .{ a, b };
            out.op = .mul;
            return out;
        }

        pub fn tanh(self: *Self, x: *Value) *Value {
            const out = self.new_value(std.math.tanh(x.data));
            out.children = .{ x, null };
            out.op = .tanh;
            return out;
        }

        pub fn zero_grads(self: *Self) void {
            var i: u32 = 0;
            while (i < self.len) : (i += 1) self.nodes[i].grad = 0;
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

                const kids = v.children;
                if (kids[0]) |k0| {
                    s1[sp1] = k0;
                    sp1 += 1;
                }
                if (kids[1]) |k1| {
                    s1[sp1] = k1;
                    sp1 += 1;
                }
            }

            self.zero_grads();
            root.grad = 1.0;

            var i: u32 = 0;
            while (i < sp2) : (i += 1) {
                const v = s2[i];
                switch (v.op orelse continue) {
                    .add => {
                        v.children[0].?.grad += v.grad;
                        v.children[1].?.grad += v.grad;
                    },
                    .mul => {
                        const a = v.children[0].?;
                        const b = v.children[1].?;
                        a.grad += b.data * v.grad;
                        b.grad += a.data * v.grad;
                    },
                    .tanh => {
                        const x = v.children[0].?;
                        const d = 1.0 - (v.data * v.data);
                        x.grad += d * v.grad;
                    },
                }
            }
        }
    };
}
