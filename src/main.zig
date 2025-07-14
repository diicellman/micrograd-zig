const std = @import("std");
const engine = @import("engine.zig");

pub fn main() !void {
    const a = engine.Value.init(2.0);
    const b = engine.Value.init(-3.0);
    const c = a.add(b);
    std.debug.print("Value(data={d:.4})\n", .{a.data});
    std.debug.print("Value(data={d:.4})\n", .{b.data});
    std.debug.print("Value(data={d:.4})\n", .{c.data});

    // i'll leave it here for now
    // std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    // const stdout_file = std.io.getStdOut().writer();
    // var bw = std.io.bufferedWriter(stdout_file);
    // const stdout = bw.writer();

    // try stdout.print("Run `zig build test` to run the tests.\n", .{});

    // try bw.flush(); // Don't forget to flush!
}
