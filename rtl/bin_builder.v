`timescale 1ns/1ps
// bin_builder.v -- event stream -> binary 2 x 32 x 32 presence map per emulator frame (model/snn_input.py: bin_emulated).
//
// Bit (pol, y >> 2, x >> 2) is set per event. At `swap` (end of an emulator frame) the bank that was being written becomes the
// bank the SNN core reads (stable until the next swap) and the other bank is cleared (2048 clocks, during which ev_ready is low:
// the emulator's event FIFO absorbs and back-pressures, nothing is lost).
// Bits can also be set through `set_*` (the clip replay path of the demo); set_valid is ignored while clearing.
module bin_builder (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        ev_valid,            // emulator event stream, ev_data = {pol, y[6:0], x[6:0], ts[16:0]}
    output wire        ev_ready,
    input  wire [31:0] ev_data,

    input  wire        set_valid,           // alternative source: set bit set_addr = {pol, y5, x5}
    input  wire [10:0] set_addr,

    input  wire        swap,                // end of frame: finished bank -> read side, clear the other
    output wire        clearing,

    input  wire [10:0] rd_addr,             // read side (SNN core), 1-clock latency
    output wire        rd_q
);
    reg         wbank;
    reg         init;                       // after reset: zero both banks (4096 clocks)
    reg [11:0]  cptr;                       // init: 0..4095; after a swap: 1..2048 (clears address cptr-1 of the new write bank)
    assign clearing = init || (cptr != 12'd0);
    assign ev_ready = !clearing;

    wire        ev_fire = ev_valid && ev_ready;
    wire [10:0] ev_addr = {ev_data[31], ev_data[30:26], ev_data[23:19]};      // {pol, y[6:2], x[6:2]}
    wire        w_set   = ev_fire || (set_valid && !clearing);
    wire [10:0] w_addr  = ev_fire ? ev_addr : set_addr;
    wire [10:0] cm1     = cptr[10:0] - 11'd1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wbank <= 1'b0; init <= 1'b1; cptr <= 12'd0;
        end else if (init) begin
            if (cptr == 12'd4095) begin init <= 1'b0; cptr <= 12'd0; end
            else cptr <= cptr + 12'd1;
        end else if (cptr != 12'd0) begin
            cptr <= (cptr == 12'd2048) ? 12'd0 : cptr + 12'd1;
        end else if (swap) begin
            wbank <= ~wbank; cptr <= 12'd1;
        end
    end

    wire        we    = clearing || w_set;
    wire [11:0] waddr = init ? cptr : (cptr != 12'd0 ? {wbank, cm1} : {wbank, w_addr});
    wire        wdata = !clearing;

    sdp_ram #(.DW(1), .AW(12)) u_map (
        .clk(clk), .we(we), .waddr(waddr), .wdata(wdata),
        .re(1'b1), .raddr({~wbank, rd_addr}), .rdata(rd_q));
endmodule
