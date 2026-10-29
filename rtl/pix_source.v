`timescale 1ns/1ps
// pix_source.v -- on-chip frame source (roadmap 1.10): a moving square on a gradient, 128x128 8-bit,
// streamed with valid/ready/sop/eop like the future frame-reader DMA. One frame every FRAME_CLKS clocks
// (833,333 at 25 MHz = 30 fps). Bit-exact to model/pattern_source.py.
//
//   frame f : square 24x24 value 200 with top-left (3f mod 128, 2f mod 128), wrapping,
//             background 40 + (x >> 2).
module pix_source #(
    parameter FRAME_CLKS = 833333
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        run,            // 0 pauses the source between frames
    output wire        pix_valid,
    input  wire        pix_ready,
    output wire [7:0]  pix_data,
    output wire        pix_sop,
    output wire        pix_eop,
    output reg  [15:0] frame_idx       // frames fully sent
);
    reg         active;
    reg [13:0]  addr;
    reg [23:0]  timer;
    reg [6:0]   sx;
    reg [5:0]   sy_h;                  // sy = 2f mod 128, bit 0 is always 0
    wire [6:0]  sy = {sy_h, 1'b0};

    wire [6:0] x = addr[6:0];
    wire [6:0] y = addr[13:7];
    wire [6:0] dx = x - sx;
    wire [6:0] dy = y - sy;
    wire in_sq = (dx < 7'd24) && (dy < 7'd24);

    assign pix_valid = active;
    assign pix_data  = in_sq ? 8'd200 : (8'd40 + {3'b000, x[6:2]});
    assign pix_sop   = active && (addr == 14'd0);
    assign pix_eop   = active && (addr == 14'd16383);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0; addr <= 14'd0; timer <= 24'd0; frame_idx <= 16'd0;
        end else if (!active) begin
            if (run) begin
                if (timer >= FRAME_CLKS - 1) begin
                    active <= 1'b1; addr <= 14'd0; timer <= 24'd0;
                end else begin
                    timer <= timer + 24'd1;
                end
            end
        end else if (pix_ready) begin
            if (addr == 14'd16383) begin
                active <= 1'b0; frame_idx <= frame_idx + 16'd1;
            end else begin
                addr <= addr + 14'd1;
            end
        end
    end

    // square position, advanced once per frame (kept apart from the stream FSM)
    wire frame_end = active && pix_ready && (addr == 14'd16383);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sx <= 7'd0; sy_h <= 6'd0;
        end else if (frame_end) begin
            sx <= sx + 7'd3; sy_h <= sy_h + 6'd1;
        end
    end
endmodule
