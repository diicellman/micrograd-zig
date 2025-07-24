const std = @import("std");
const engine = @import("engine.zig");

pub fn main() !void {
    var a: engine.Value = undefined;
    a.init(2.0);
    var b: engine.Value = undefined;
    b.init(-3.0);
    var c: engine.Value = undefined;
    c.init(10.0);

    const temp = try a.mul(&b);
    const result = try temp.add(&c);

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
