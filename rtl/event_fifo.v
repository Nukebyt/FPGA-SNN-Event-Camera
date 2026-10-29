`timescale 1ns/1ps
// event_fifo.v -- synchronous FIFO, valid/ready output, M10K storage.
//
//   * Write side has no ready: the producer must keep `level` below DEPTH (the emulator throttles
//     its pixel input on `level`). A write while full is NOT stored and pulses `overflow`, so a
//     lost event is always visible (roadmap 4.6: never silently drop).
//   * The RAM read register is the output register: `rd_data` is stable while `rd_valid && !rd_ready`.
//   * `level` counts entries in the RAM only (the output register holds one more).
//   * Throughput: one word per clock in both directions.
module event_fifo #(
    parameter AW = 10,
    parameter DW = 32
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          wr_en,
    input  wire [DW-1:0] wr_data,
    output wire [AW:0]   level,
    output reg           overflow,   // 1-cycle pulse
    output wire [DW-1:0] rd_data,
    output reg           rd_valid,
    input  wire          rd_ready
);
    reg [AW:0]   cnt;
    reg [AW-1:0] wp, rp;

    wire full   = cnt[AW];                               // cnt == 2**AW
    wire do_wr  = wr_en && !full;
    wire issue  = (cnt != 0) && (!rd_valid || rd_ready);

    assign level = cnt;

    sdp_ram #(.DW(DW), .AW(AW)) u_ram (
        .clk(clk), .we(do_wr), .waddr(wp), .wdata(wr_data),
        .re(issue), .raddr(rp), .rdata(rd_data)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt <= 0; wp <= 0; rp <= 0; rd_valid <= 1'b0; overflow <= 1'b0;
        end else begin
            overflow <= wr_en && full;
            if (do_wr) wp <= wp + 1'b1;
            if (issue) rp <= rp + 1'b1;
            cnt <= cnt + (do_wr ? 1 : 0) - (issue ? 1 : 0);
            if (issue)         rd_valid <= 1'b1;
            else if (rd_ready) rd_valid <= 1'b0;
        end
    end
endmodule
