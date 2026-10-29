`timescale 1ns/1ps
// snn_demo_top_tb.v -- the board top level as the user will see it: KEY/SW in, HEX/LEDR out.
//   REPLAY (default): select clip NPRESS with KEY[3], wait for the first decision window, and require that HEX0 shows the class the
//                     Python model predicts for that clip window (clips_pred.hex), HEX2 the clip's true class, HEX3 = 1 window,
//                     exactly one LEDR class light, no error LED.
//   LIVE (-DLIVE=1):  pattern source -> RTL emulator -> SNN with a short window (-DTPAR), HEX0 must show the model's class
//                     for that many pattern frames (expect_final.hex of `gen_snn_vectors.py chain`).
// Macros: -DVEC="dir" (weights, clips, expectations), -DNPRESS=n, -DTPAR=n, -DSRC_CLKS=n, -DLIVE=1
`ifndef NPRESS
`define NPRESS 0
`endif
`ifndef TPAR
`define TPAR 32
`endif
`ifndef SRC_CLKS
`define SRC_CLKS 262144
`endif
`ifndef LIVE
`define LIVE 0
`endif
module snn_demo_top_tb;
    reg        CLOCK_50 = 1'b0;
    always #10 CLOCK_50 = ~CLOCK_50;
    reg  [3:0] KEY = 4'b1110;
    reg  [9:0] SW  = (`LIVE ? 10'b10_0000_0100 : 10'b00_0000_0100);
    wire [9:0] LEDR;
    wire [6:0] H0, H1, H2, H3, H4, H5;
    wire       VGA_CLK, VGA_HS, VGA_VS, VGA_BLANK_N, VGA_SYNC_N;
    wire [7:0] VGA_R, VGA_G, VGA_B;

    snn_demo_top #(.FRAME_CLKS(`SRC_CLKS), .T(`TPAR),
                   .W1FILE({`VEC, "/w1.hex"}), .W2FILE({`VEC, "/w2.hex"}), .WFFILE({`VEC, "/wfc.hex"}),
                   .CLIPFILE({`VEC, "/clips.hex"}), .LBLFILE({`VEC, "/labels.hex"})) dut (
        .CLOCK_50(CLOCK_50), .KEY(KEY), .SW(SW), .LEDR(LEDR),
        .HEX0(H0), .HEX1(H1), .HEX2(H2), .HEX3(H3), .HEX4(H4), .HEX5(H5),
        .VGA_CLK(VGA_CLK), .VGA_HS(VGA_HS), .VGA_VS(VGA_VS), .VGA_R(VGA_R), .VGA_G(VGA_G),
        .VGA_B(VGA_B), .VGA_BLANK_N(VGA_BLANK_N), .VGA_SYNC_N(VGA_SYNC_N));

    // digit -> active-low segments, same table as rtl/hex7seg.v (kept independent on purpose)
    function [6:0] seg;
        input [3:0] d;
        case (d)
            4'h0: seg = 7'b1000000; 4'h1: seg = 7'b1111001; 4'h2: seg = 7'b0100100; 4'h3: seg = 7'b0110000;
            4'h4: seg = 7'b0011001; 4'h5: seg = 7'b0010010; 4'h6: seg = 7'b0000010; 4'h7: seg = 7'b1111000;
            4'h8: seg = 7'b0000000; 4'h9: seg = 7'b0010000; 4'hA: seg = 7'b0001000; 4'hB: seg = 7'b0000011;
            4'hC: seg = 7'b1000110; 4'hD: seg = 7'b0100001; 4'hE: seg = 7'b0000110; 4'hF: seg = 7'b0001110;
        endcase
    endfunction

    reg [3:0] expc [0:63];          // clips_pred.hex: 2 windows per clip
    reg [3:0] lbl  [0:7];
    reg [31:0] expf [0:15];         // live: expect_final.hex (class first)
    integer i, errs, p;
    reg [3:0] want, want_lbl;
    initial begin
        errs = 0;
        for (i = 0; i < 64; i = i + 1) expc[i] = 4'hF;
        for (i = 0; i < 8; i = i + 1) lbl[i] = 4'hF;
        if (`LIVE) $readmemh({`VEC, "/expect_final.hex"}, expf);
        else begin
            $readmemh({`VEC, "/clips_pred.hex"}, expc);
            $readmemh({`VEC, "/labels.hex"}, lbl);
        end
        #200 KEY[0] = 1'b1;
        repeat (6000) @(posedge CLOCK_50);                // builder init sweep and the first core clear
        for (p = 0; p < `NPRESS; p = p + 1) begin
            KEY[3] = 1'b0; repeat (40) @(posedge CLOCK_50);
            KEY[3] = 1'b1; repeat (6000) @(posedge CLOCK_50);
        end
        wait (dut.u_ctl.windows == 16'd1);
        repeat (20) @(posedge CLOCK_50);
        want = `LIVE ? expf[0][3:0] : expc[`NPRESS * 2];
        want_lbl = `LIVE ? 4'hF : lbl[`NPRESS];
        if (H0 !== seg(want))      begin errs = errs + 1; $display("MISMATCH HEX0 shows %b, model class %0d (%b)", H0, want, seg(want)); end
        if (H1 !== seg(`LIVE ? 4'hF : `NPRESS)) begin errs = errs + 1; $display("MISMATCH HEX1 clip number"); end
        if (H2 !== seg(want_lbl))  begin errs = errs + 1; $display("MISMATCH HEX2 shows %b, expected true class %0d", H2, want_lbl); end
        if (H3 !== seg(4'd1))      begin errs = errs + 1; $display("MISMATCH HEX3 (windows) %b", H3); end
        if (LEDR[9:4] !== (6'd1 << want)) begin errs = errs + 1; $display("MISMATCH class LEDs %b, expected one-hot %0d", LEDR[9:4], want); end
        if (LEDR[3])               begin errs = errs + 1; $display("MISMATCH error LED lit"); end
        if (dut.u_ctl.pred !== want) begin errs = errs + 1; $display("MISMATCH controller class %0d, model %0d", dut.u_ctl.pred, want); end
        $display("info: %s, clip %0d, window 1: board shows class %0d (model %0d), %0d errors, emulator frames %0d",
                 `LIVE ? "LIVE" : "REPLAY", `NPRESS, dut.u_ctl.pred, want, errs, dut.frame_cnt);
        if (errs == 0) $display("RESULT: PASS");
        else           $display("RESULT: FAIL");
        $finish;
    end

    initial begin
        #4000000000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end
endmodule
