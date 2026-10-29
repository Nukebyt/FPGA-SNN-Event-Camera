`timescale 1ns/1ps
// frame_sync.v -- one `frame_done` pulse per finished emulator frame, once its events have all been written to the bin builder.
// The emulator's frame counter increments when the last pixel of a frame is accepted; the frame is finished when the emulator is
// idle again (pixel pipeline and event FIFO empty). Two extra idle clocks let the builder's last write land.
module frame_sync (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        emu_idle,
    input  wire [31:0] frame_cnt,
    output reg         frame_done
);
    reg [31:0] fc_d;
    reg        pend, idle_d, idle_dd;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc_d <= 32'd0; pend <= 1'b0; idle_d <= 1'b0; idle_dd <= 1'b0; frame_done <= 1'b0;
        end else begin
            fc_d <= frame_cnt; idle_d <= emu_idle; idle_dd <= idle_d;
            frame_done <= 1'b0;
            if (frame_cnt == fc_d + 32'd1) pend <= 1'b1;        // not on a counter clear (KEY2), only on a real +1
            else if (pend && emu_idle && idle_d && idle_dd) begin pend <= 1'b0; frame_done <= 1'b1; end
        end
    end
endmodule
