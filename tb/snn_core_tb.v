`timescale 1ns/1ps
// snn_core_tb.v -- bit-exactness of bin_builder + snn_core + snn_ctrl against model/snn_int.py.
// Bins are set through the builder's set_* port, one emulator frame at a time; after every step the layer spike counts and the class
// currents are compared with the model's, after the last step the window result (argmax, summed scores) as well.
// Vectors: scripts/gen_snn_vectors.py (params.vh, bins.hex, expect_steps.hex, expect_final.hex, w1/w2/wfc.hex in `VEC).
`include "params.vh"
module snn_core_tb;
    localparam NC = `NCLS;
    reg clk = 1'b0, rst_n = 1'b0;
    always #20 clk = ~clk;

    reg         set_valid = 1'b0;
    reg  [10:0] set_addr  = 11'd0;
    reg         swap      = 1'b0;
    wire        clearing, in_q, core_busy, core_done, core_start, core_clear, overrun;
    wire [10:0] in_addr;
    wire [NC*28-1:0] score_flat, cur_flat, win_score;
    wire [19:0] spk1, spk2;
    wire [3:0]  pred;
    wire [15:0] windows;
    wire [7:0]  wcount;

    bin_builder u_b (.clk(clk), .rst_n(rst_n), .ev_valid(1'b0), .ev_ready(), .ev_data(32'd0),
                     .set_valid(set_valid), .set_addr(set_addr), .swap(swap), .clearing(clearing),
                     .rd_addr(in_addr), .rd_q(in_q));
    snn_core #(.C1(`C1), .C2(`C2), .NCLS(NC), .W1FILE({`VEC, "/w1.hex"}), .W2FILE({`VEC, "/w2.hex"}), .WFFILE({`VEC, "/wfc.hex"})) u_c (
        .clk(clk), .rst_n(rst_n), .start(core_start), .clear(core_clear), .busy(core_busy), .done(core_done),
        .in_addr(in_addr), .in_q(in_q), .score_flat(score_flat), .cur_flat(cur_flat), .spk1(spk1), .spk2(spk2));
    snn_ctrl #(.T(`NSTEP), .NCLS(NC)) u_t (
        .clk(clk), .rst_n(rst_n), .step_trig(swap), .restart(1'b0), .clear_flags(1'b0),
        .core_start(core_start), .core_clear(core_clear), .core_busy(core_busy), .core_done(core_done), .score_flat(score_flat),
        .pred(pred), .win_score(win_score), .windows(windows), .wcount(wcount), .overrun(overrun));

    reg [31:0] binw   [0:`NSTEP*64-1];
    reg [31:0] exps   [0:`NSTEP*(NC+2)-1];
    reg [31:0] expf   [0:NC];
    reg [95:0] ev1    [0:2047];
    reg [95:0] ev2    [0:1023];
    initial begin
        $readmemh({`VEC, "/bins.hex"}, binw);
        $readmemh({`VEC, "/expect_steps.hex"}, exps);
        $readmemh({`VEC, "/expect_final.hex"}, expf);
        $readmemh({`VEC, "/expect_v1.hex"}, ev1);
        $readmemh({`VEC, "/expect_v2.hex"}, ev2);
    end

    integer step, w, b, k, a, errs, checked, cyc_start, step_cyc;
    reg [31:0] got;
    reg done_seen;

    // per-step check at every core_done of a step (not of a clear)
    integer ck_step = 0;
    reg in_step = 1'b0;
    always @(posedge clk) begin
        if (core_start) in_step <= 1'b1;
        if (core_done && in_step) begin
            in_step <= 1'b0;
            if (spk1 !== exps[ck_step*(NC+2)])     begin errs = errs + 1; $display("MISMATCH step %0d spk1 got %0d exp %0d", ck_step, spk1, exps[ck_step*(NC+2)]); end
            if (spk2 !== exps[ck_step*(NC+2)+1])   begin errs = errs + 1; $display("MISMATCH step %0d spk2 got %0d exp %0d", ck_step, spk2, exps[ck_step*(NC+2)+1]); end
            for (k = 0; k < NC; k = k + 1) begin
                got = {{4{cur_flat[k*28+27]}}, cur_flat[k*28 +: 28]};
                if (got !== exps[ck_step*(NC+2)+2+k]) begin
                    errs = errs + 1; $display("MISMATCH step %0d class %0d current got %0d exp %0d", ck_step, k, $signed(got), $signed(exps[ck_step*(NC+2)+2+k]));
                end
            end
            if (ck_step == `NSTEP - 1) begin     // membranes after the last step (the window-end clear has not started yet)
                for (a = 0; a < 2048; a = a + 1) if (u_c.u_l1.u_mem.mem[a] !== ev1[a]) begin
                    errs = errs + 1; if (errs < 12) $display("MISMATCH layer 1 membrane word %0d got %h exp %h", a, u_c.u_l1.u_mem.mem[a], ev1[a]);
                end
                for (a = 0; a < 1024; a = a + 1) if (u_c.u_l2.u_mem.mem[a] !== ev2[a]) begin
                    errs = errs + 1; if (errs < 12) $display("MISMATCH layer 2 membrane word %0d got %h exp %h", a, u_c.u_l2.u_mem.mem[a], ev2[a]);
                end
            end
            checked = checked + 1;
            $display("info: step %0d spikes l1 %0d l2 %0d, %0d clocks", ck_step, spk1, spk2, $time / 40 - cyc_start);
            ck_step = ck_step + 1;
        end
    end

    initial begin
        errs = 0; checked = 0; cyc_start = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        // builder init sweep (4096 clocks) and the initial core clear
        wait (!clearing);
        repeat (4) @(posedge clk);
        wait (!core_busy);
        repeat (4) @(posedge clk);
        for (step = 0; step < `NSTEP; step = step + 1) begin
            wait (!clearing && !core_busy);
            @(posedge clk);
            for (w = 0; w < 64; w = w + 1)
                for (b = 0; b < 32; b = b + 1)
                    if (binw[step*64 + w][b]) begin
                        set_valid = 1'b1; set_addr = {w[5], w[4:0], b[4:0]};
                        @(posedge clk);
                    end
            set_valid = 1'b0;
            @(posedge clk);
            cyc_start = $time / 40;
            swap = 1'b1; @(posedge clk); swap = 1'b0;
            wait (core_busy);
            wait (!core_busy);
            repeat (3) @(posedge clk);
        end
        // window result (latched when the last step finished)
        repeat (10) @(posedge clk);
        if (windows !== 16'd1) begin errs = errs + 1; $display("MISMATCH windows %0d (expected 1)", windows); end
        if ({28'd0, pred} !== expf[0]) begin errs = errs + 1; $display("MISMATCH predicted class %0d expected %0d", pred, expf[0]); end
        for (k = 0; k < NC; k = k + 1) begin
            got = {{4{win_score[k*28+27]}}, win_score[k*28 +: 28]};
            if (got !== expf[1+k]) begin errs = errs + 1; $display("MISMATCH window score %0d got %0d exp %0d", k, $signed(got), $signed(expf[1+k])); end
        end
        if (overrun) begin errs = errs + 1; $display("MISMATCH overrun flag set"); end
        $display("info: %0d steps checked, window class %0d, %0d errors", checked, pred, errs);
        if (errs == 0 && checked == `NSTEP) $display("RESULT: PASS");
        else $display("RESULT: FAIL");
        $finish;
    end

    initial begin
        #2000000000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end
endmodule
