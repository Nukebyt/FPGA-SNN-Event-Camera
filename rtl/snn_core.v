`timescale 1ns/1ps
// snn_core.v -- integer spiking network inference, bit-exact to model/snn_int.py (two conv layers, sum readout):
//   input 2 x 32 x 32 bits -> conv1 (2 -> C1) LIF -> OR-pool -> conv2 (C1 -> C2) LIF -> OR-pool -> FC -> NCLS class currents.
// One `start` processes one time step (one emulator frame). The class currents of every step are added to score[];
// `clear` zeroes all membranes and scores (start of a decision window).
//
// The input bit map is read through (in_addr, in_q): address {channel, y, x} (channel 0 = OFF, 1 = ON), 1-clock read latency.
// It is owned by bin_builder, which keeps the finished bank stable while busy is high.
//
// Time per step at the clock rate f: (C1/8*1024*22 + C2/8*256*148 + NCLS*C2*64) clocks, i.e. 49k + 151k + 12k = 212k clocks
// for 16/32 channels and 6 classes (8.5 ms at 25 MHz, a quarter of the 33 ms frame).
module snn_core #(
    parameter C1     = 16,
    parameter C2     = 32,
    parameter NCLS   = 6,
    parameter THR1   = 64,
    parameter THR2   = 64,
    parameter LEAK   = 4,
    parameter W1FILE = "rtl/snn_w1.hex",
    parameter W2FILE = "rtl/snn_w2.hex",
    parameter WFFILE = "rtl/snn_wfc.hex"
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        clear,
    output wire        busy,
    output reg         done,                 // pulse: step (or clear) finished

    output wire [10:0] in_addr,
    input  wire        in_q,

    output wire [NCLS*28-1:0] score_flat,    // summed class currents since the last clear, signed 28 bit each
    output wire [NCLS*28-1:0] cur_flat,      // class currents of the last step
    output reg  [19:0] spk1,                 // layer 1 spikes of the last step (before pooling)
    output reg  [19:0] spk2
);
    localparam G1  = C1 / 8;
    localparam G2  = C2 / 8;
    localparam FCIN = C2 * 64;               // 2048 for C2 = 32
    localparam FCAW = $clog2(NCLS * FCIN);
    localparam IW  = $clog2(FCIN);
    localparam P1W = $clog2(G1 * 256);
    localparam P2W = $clog2(G2 * 64);

    // ---------------- layer 1 ----------------
    wire         l1_busy, l1_done;
    wire [4:0]   l1_ci;
    wire [5:0]   l1_y, l1_x;
    wire [P1W-1:0] l1_nx_addr;
    wire [7:0]   l1_nx_q;
    wire [19:0]  l1_spk;
    reg          l1_start, l1_clear;
    assign in_addr = {l1_ci[0], l1_y[4:0], l1_x[4:0]};
    lif_conv #(.CIN(2), .GROUPS(G1), .HW(32), .THR(THR1), .LEAK(LEAK), .WFILE(W1FILE)) u_l1 (
        .clk(clk), .rst_n(rst_n), .start(l1_start), .clear(l1_clear), .busy(l1_busy), .done(l1_done),
        .rd_ci(l1_ci), .rd_y(l1_y), .rd_x(l1_x), .rd_bit(in_q),
        .nx_addr(l1_nx_addr), .nx_q(l1_nx_q), .spk_cnt(l1_spk));

    // ---------------- layer 2 (input = pooled layer 1 words) ----------------
    wire         l2_busy, l2_done;
    wire [4:0]   l2_ci;
    wire [5:0]   l2_y, l2_x;
    wire [P2W-1:0] l2_nx_addr;
    wire [7:0]   l2_nx_q;
    wire [19:0]  l2_spk;
    reg          l2_start, l2_clear;
    reg  [2:0]   l2_ci_lo_d;
    assign l1_nx_addr = ((l2_ci >> 3) * 256) + (l2_y[3:0] * 16) + l2_x[3:0];
    always @(posedge clk) l2_ci_lo_d <= l2_ci[2:0];
    wire l2_bit = l1_nx_q[l2_ci_lo_d];
    lif_conv #(.CIN(C1), .GROUPS(G2), .HW(16), .THR(THR2), .LEAK(LEAK), .WFILE(W2FILE)) u_l2 (
        .clk(clk), .rst_n(rst_n), .start(l2_start), .clear(l2_clear), .busy(l2_busy), .done(l2_done),
        .rd_ci(l2_ci), .rd_y(l2_y), .rd_x(l2_x), .rd_bit(l2_bit),
        .nx_addr(l2_nx_addr), .nx_q(l2_nx_q), .spk_cnt(l2_spk));

    // ---------------- classifier: NCLS x FCIN int8 weights, input order (group, y, x, channel-in-group) ----------------
    reg signed [7:0] fc_rom [0:NCLS*FCIN-1];
    initial $readmemh(WFFILE, fc_rom);
    reg signed [7:0] fc_q;
    reg [FCAW-1:0]   fn;                     // running index k*FCIN + i
    reg [IW-1:0]     fi;
    reg [3:0]        fk;
    always @(posedge clk) fc_q <= fc_rom[fn];
    assign l2_nx_addr = fi >> 3;

    reg f_run, f1_valid, f1_last_i;
    reg [2:0] f1_j;
    reg [3:0] f1_k;
    reg signed [27:0] facc;
    reg signed [27:0] score [0:NCLS-1];
    reg signed [27:0] cur   [0:NCLS-1];
    wire signed [27:0] term   = (l2_nx_q[f1_j]) ? {{20{fc_q[7]}}, fc_q} : 28'sd0;
    wire signed [27:0] sumnow = facc + term;
    wire f_last_class = (f1_k == NCLS - 1);

    genvar gi;
    generate
        for (gi = 0; gi < NCLS; gi = gi + 1) begin : g_out
            assign score_flat[gi*28 +: 28] = score[gi];
            assign cur_flat[gi*28 +: 28]   = cur[gi];
        end
    endgenerate

    // ---------------- sequencing ----------------
    localparam [2:0] C_IDLE = 3'd0, C_CLR = 3'd1, C_L1 = 3'd2, C_L2 = 3'd3, C_FC = 3'd4;
    reg [2:0] cs;
    reg       clr1_done, clr2_done;
    assign busy = (cs != C_IDLE);
    integer s;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cs <= C_IDLE; done <= 1'b0; l1_start <= 1'b0; l1_clear <= 1'b0; l2_start <= 1'b0; l2_clear <= 1'b0;
            clr1_done <= 1'b0; clr2_done <= 1'b0; fn <= 0; fi <= 0; fk <= 4'd0; f_run <= 1'b0; f1_valid <= 1'b0;
            f1_last_i <= 1'b0; f1_j <= 3'd0; f1_k <= 4'd0; facc <= 28'sd0; spk1 <= 20'd0; spk2 <= 20'd0;
            for (s = 0; s < NCLS; s = s + 1) begin score[s] <= 28'sd0; cur[s] <= 28'sd0; end
        end else begin
            done <= 1'b0; l1_start <= 1'b0; l1_clear <= 1'b0; l2_start <= 1'b0; l2_clear <= 1'b0;

            // classifier issue / accumulate pipeline
            f1_valid  <= f_run;
            f1_j      <= fi[2:0];
            f1_k      <= fk;
            f1_last_i <= (fi == FCIN - 1);
            if (f_run) begin
                fn <= fn + 1'b1;
                if (fi == FCIN - 1) begin
                    fi <= 0; fk <= fk + 1'b1;
                end else fi <= fi + 1'b1;
                if (fn == NCLS * FCIN - 1) f_run <= 1'b0;
            end
            if (f1_valid) begin
                if (f1_last_i) begin
                    score[f1_k] <= score[f1_k] + sumnow;
                    cur[f1_k]   <= sumnow;
                    facc <= 28'sd0;
                end else facc <= sumnow;
            end

            case (cs)
                C_IDLE: begin
                    if (start) begin
                        l1_start <= 1'b1; cs <= C_L1;
                    end else if (clear) begin
                        l1_clear <= 1'b1; l2_clear <= 1'b1; clr1_done <= 1'b0; clr2_done <= 1'b0;
                        for (s = 0; s < NCLS; s = s + 1) begin score[s] <= 28'sd0; cur[s] <= 28'sd0; end
                        cs <= C_CLR;
                    end
                end
                C_CLR: begin
                    if (l1_done) clr1_done <= 1'b1;
                    if (l2_done) clr2_done <= 1'b1;
                    if ((clr1_done || l1_done) && (clr2_done || l2_done)) begin cs <= C_IDLE; done <= 1'b1; end
                end
                C_L1: if (l1_done) begin
                    spk1 <= l1_spk; l2_start <= 1'b1; cs <= C_L2;
                end
                C_L2: if (l2_done) begin
                    spk2 <= l2_spk;
                    fn <= 0; fi <= 0; fk <= 4'd0; facc <= 28'sd0; f_run <= 1'b1; cs <= C_FC;
                end
                C_FC: if (f1_valid && f1_last_i && f_last_class) begin
                    cs <= C_IDLE; done <= 1'b1;
                end
                default: cs <= C_IDLE;
            endcase
        end
    end
endmodule
