`timescale 1ns/1ps
// sdp_ram.v -- simple dual-port RAM, one clock, registered read, no read-during-write guarantee.
// Standalone module (not inside a generate) so Quartus infers M10K (checklist 1.1).
// Users must never read and write the same address in the same cycle:
//   * reference memory: a pixel is read again one whole frame later,
//   * event FIFO: the read side only touches entries already counted as written.
module sdp_ram #(
    parameter DW = 8,
    parameter AW = 14
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [DW-1:0] wdata,
    input  wire          re,
    input  wire [AW-1:0] raddr,
    output reg  [DW-1:0] rdata
);
    (* ramstyle = "M10K, no_rw_check" *) reg [DW-1:0] mem [0:(1<<AW)-1];
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        if (re) rdata <= mem[raddr];
    end
endmodule
