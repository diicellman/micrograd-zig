const std = @import("std");

const OpType = enum {
    add,
    mul,
};

pub const Value = struct {
    data: f32,
    grad: f32,
    children: [2]?*const Value,
    op: ?OpType,

    pub fn init(self: *Value, data: f32) void {
        self.* = .{
            .data = data,
            .grad = 0.0,
            .children = .{ null, null },
            .op = null,
        };
    }

    pub fn add(self: *const Value, other: *const Value) !Value {
        const result = Value{ .data = self.data + other.data, .grad = 0.0, .children = .{ self, other }, .op = OpType.add };

        return result;
    }

    pub fn mul(self: *const Value, other: *const Value) !Value {
        const result = Value{ .data = self.data + other.data, .grad = 0.0, .children = .{ self, other }, .op = OpType.add };
        return result;
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
