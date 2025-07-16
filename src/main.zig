const std = @import("std");
const engine = @import("engine.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();

    const allocator = arena.allocator();

    const a = engine.Value.init(2.0);
    const b = engine.Value.init(-3.0);
    const c = engine.Value.init(10.0);

    const temp = try a.mul(&b, allocator);
    const result = try temp.add(&c, allocator);

    a.debug();
    b.debug();
    c.debug();
    result.debug();
    // std.debug.print("the result is: {d:.4}\n", .{result.data});

    // i'll leave it here for now
    // std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    // const stdout_file = std.io.getStdOut().writer();
    // var bw = std.io.bufferedWriter(stdout_file);
    // const stdout = bw.writer();

    // try stdout.print("Run `zig build test` to run the tests.\n", .{});

    // try bw.flush(); // Don't forget to flush!
}
