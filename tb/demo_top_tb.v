`timescale 1ns/1ps
// demo_top_tb.v -- runs the on-chip demo and dumps complete VGA frames for scripts/check_demo_vga.py,
// which compares them pixel for pixel with the model (pattern -> emulator_int -> event view).
// Parameters come from the command line as macros:
//   -DSWVAL=10'bxxxxxxxxxx  -DDUMP_DIR="work/demo"  -DNDUMP=n
`ifndef SWVAL
`define SWVAL 10'b00_0000_0100
`endif
`ifndef DUMP_DIR
`define DUMP_DIR "work/demo"
`endif
`ifndef NDUMP
`define NDUMP 6
`endif
`ifndef SRC_CLKS
`define SRC_CLKS 100000
`endif

module demo_top_tb;
    reg        CLOCK_50 = 1'b0;
    always #10 CLOCK_50 = ~CLOCK_50;
    reg  [3:0] KEY = 4'b1110;               // KEY0 pressed (reset) at start
    reg  [9:0] SW  = `SWVAL;
    wire [9:0] LEDR;
    wire [6:0] H0, H1, H2, H3, H4, H5;
    wire       VGA_CLK, VGA_HS, VGA_VS, VGA_BLANK_N, VGA_SYNC_N;
    wire [7:0] VGA_R, VGA_G, VGA_B;

    demo_top #(.FRAME_CLKS(`SRC_CLKS)) dut (
        .CLOCK_50(CLOCK_50), .KEY(KEY), .SW(SW), .LEDR(LEDR),
        .HEX0(H0), .HEX1(H1), .HEX2(H2), .HEX3(H3), .HEX4(H4), .HEX5(H5),
        .VGA_CLK(VGA_CLK), .VGA_HS(VGA_HS), .VGA_VS(VGA_VS), .VGA_R(VGA_R), .VGA_G(VGA_G),
        .VGA_B(VGA_B), .VGA_BLANK_N(VGA_BLANK_N), .VGA_SYNC_N(VGA_SYNC_N));

    reg [23:0] img [0:307199];
    integer    idx = 0;
    integer    nframe = 0;
    reg [8*80-1:0] fname;
    reg [15:0] first_l, last_l;
    reg        vs_d = 1'b1;
    integer    bad = 0;
    reg        started = 1'b0;

    initial begin
        #200 KEY[0] = 1'b1;
    end

    // one process owns all capture state
    always @(posedge VGA_CLK) begin
        vs_d <= VGA_VS;
        if (VGA_BLANK_N) begin
            if (idx == 0) begin first_l = dut.u_view.disp_first; last_l = dut.u_view.disp_last; end
            img[idx] <= {VGA_R, VGA_G, VGA_B};
            idx = idx + 1;
        end
        if (vs_d && !VGA_VS) begin                      // start of vertical sync = end of the frame's active area
            if (idx == 307200 && started) begin
                $sformat(fname, "%0s/frame%02d.hex", `DUMP_DIR, nframe);
                $writememh(fname, img);
                $display("FRAME %0d first=%0d last=%0d src_frames=%0d frame_cnt=%0d err=%b ovf=%0d",
                         nframe, first_l, last_l, dut.src_frames, dut.frame_cnt, dut.err_flags, dut.ev_overflow_cnt);
                nframe = nframe + 1;
                if (dut.err_flags != 0 || dut.ev_overflow_cnt != 0) bad = bad + 1;
                if (nframe >= `NDUMP) begin
                    if (bad == 0) $display("RESULT: PASS (frames dumped; compare with check_demo_vga.py)");
                    else          $display("RESULT: FAIL (error flags set)");
                    $finish;
                end
            end else if (idx != 0 && started) begin
                $display("FAIL: VGA frame had %0d active pixels, expected 307200", idx); bad = bad + 1;
            end
            started = 1'b1;
            idx = 0;
        end
    end

    initial begin
        #400_000_000;                                    // 400 ms of simulated time
        $display("RESULT: FAIL (timeout)");
        $finish;
    end
endmodule
