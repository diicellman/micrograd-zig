const std = @import("std");
const engine = @import("engine.zig");

pub fn Neuron(comptime IN: usize, comptime MAXN: u32) type {
    return struct {
        const Self = @This();

        tape: *engine.Tape(MAXN),
        w: [IN]*engine.Value,
        b: *engine.Value,

        pub fn init(self: *Self, tape: *engine.Tape(MAXN), rng: anytype) void {
            self.tape = tape;
            inline for (0..IN) |i| {
                const r: f32 = @floatCast(rng.float(f64) * 2.0 - 1.0);
                self.w[i] = tape.new_value(r);
            }
            self.b = tape.new_value(0.0);
        }

        pub fn forward(self: *Self, x: [IN]*engine.Value) *engine.Value {
            // sum = x·w + b
            var acc = self.tape.add(self.tape.mul(x[0], self.w[0]), self.b);
            inline for (1..IN) |i| {
                const prod = self.tape.mul(x[i], self.w[i]);
                acc = self.tape.add(acc, prod);
            }
            return self.tape.tanh(acc);
        }

        pub fn for_each_param(self: *Self, cb: anytype) void {
            inline for (self.w) |p| cb(p);
            cb(self.b);
        }
    };
}

pub fn Layer(comptime IN: usize, comptime OUT: usize, comptime MAXN: u32) type {
    return struct {
        const Self = @This();
        neurons: [OUT]Neuron(IN, MAXN),

        pub fn init(self: *Self, tape: *engine.Tape(MAXN), rng: anytype) void {
            inline for (0..OUT) |i| self.neurons[i].init(tape, rng);
        }

        pub fn forward(self: *Self, x: [IN]*engine.Value) [OUT]*engine.Value {
            var y: [OUT]*engine.Value = undefined;
            inline for (0..OUT) |i| y[i] = self.neurons[i].forward(x);
            return y;
        }

        pub fn for_each_param(self: *Self, cb: anytype) void {
            inline for (0..OUT) |i| self.neurons[i].for_each_param(cb);
        }
    };
}

pub fn MLP(
    comptime IN: usize,
    comptime H1: usize,
    comptime H2: usize,
    comptime OUT: usize,
    comptime MAXN: u32,
) type {
    return struct {
        const Self = @This();
        l1: Layer(IN, H1, MAXN),
        l2: Layer(H1, H2, MAXN),
        l3: Layer(H2, OUT, MAXN),

        pub fn init(self: *Self, tape: *engine.Tape(MAXN), rng: anytype) void {
            self.l1.init(tape, rng);
            self.l2.init(tape, rng);
            self.l3.init(tape, rng);
        }

        pub fn forward(self: *Self, x: [IN]*engine.Value) [OUT]*engine.Value {
            const h1 = self.l1.forward(x);
            const h2 = self.l2.forward(h1);
            return self.l3.forward(h2);
        }

        pub fn for_each_param(self: *Self, cb: anytype) void {
            self.l1.for_each_param(cb);
            self.l2.for_each_param(cb);
            self.l3.for_each_param(cb);
        }
    };
}

pub fn sgd_step(params_cb: anytype, lr: f32) void {
    params_cb(struct {
        fn cb(p: *engine.Value) void {
            p.data -= lr * p.grad;
        }
    }.cb);
}
