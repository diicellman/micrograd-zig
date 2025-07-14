const std = @import("std");

pub const Value = struct {
    data: f32,

    pub fn init(data: f32) Value {
        return Value{ .data = data };
    }

    pub fn add(self: Value, other: Value) Value {
        return Value{ .data = self.data + other.data };
    }
};
