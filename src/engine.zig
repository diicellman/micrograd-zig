const std = @import("std");

pub const Value = struct {
    data: f32,
    children: std.ArrayListUnmanaged(*const Value),

    pub fn init(data: f32) Value {
        return Value{
            .data = data,
            .children = .empty,
        };
    }

    pub fn add(self: *const Value, other: *const Value, allocator: std.mem.Allocator) !Value {
        var result = Value{ .data = self.data + other.data, .children = .empty };
        try result.children.ensureUnusedCapacity(allocator, 2);
        try result.children.append(allocator, self);
        try result.children.append(allocator, other);
        return result;
    }

    pub fn mul(self: *const Value, other: *const Value, allocator: std.mem.Allocator) !Value {
        var result = Value{ .data = self.data * other.data, .children = .empty };
        try result.children.ensureUnusedCapacity(allocator, 2);
        try result.children.append(allocator, self);
        try result.children.append(allocator, other);
        return result;
    }

    // very shaky impl, maybe the future me will regret this
    pub fn debug(self: Value) void {
        std.debug.print("Value(data={d:.4}", .{self.data});
        if (self.children.items.len > 0) {
            std.debug.print(", children=[", .{});
            for (self.children.items, 0..) |child, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{d:.4}", .{child.data});
            }
            std.debug.print("]", .{});
        }

        std.debug.print(")\n", .{});
    }
};
