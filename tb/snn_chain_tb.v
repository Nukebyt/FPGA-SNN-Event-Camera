`timescale 1ns/1ps
// snn_chain_tb.v -- the whole RTL chain against the Python models: pix_source -> event_emulator -> frame_sync -> bin_builder ->
// snn_core -> snn_ctrl. Per-step layer spike counts and class currents, and the window result, must equal
// model: pattern_frame -> emulate (theta 9, step) -> bin_emulated -> IntSNN (scripts/gen_snn_vectors.py chain).
`include "params.vh"
`ifndef SRC_CLKS
`define SRC_CLKS 260000
`endif
module snn_chain_tb;
    localparam NC = `NCLS;
    reg clk = 1'b0, rst_n = 1'b0;
    always #20 clk = ~clk;

    wire        pix_valid, pix_ready, pix_sop, pix_eop;
    wire [7:0]  pix_data;
    wire [15:0] src_frames;
    pix_source #(.FRAME_CLKS(`SRC_CLKS)) u_src (.clk(clk), .rst_n(rst_n), .run(1'b1), .pix_valid(pix_valid), .pix_ready(pix_ready),
        .pix_data(pix_data), .pix_sop(pix_sop), .pix_eop(pix_eop), .frame_idx(src_frames));
    wire        ev_valid, ev_ready, emu_idle;
    wire [31:0] ev_data, frame_cnt, ev_gen_cnt;
    wire [3:0]  err_flags;
    wire [15:0] ev_overflow_cnt;
    wire [10:0] fifo_level;
    event_emulator #(.W(128), .H(128), .FIFO_AW(10), .LUT_FILE("rtl/log_lut.hex")) u_emu (
        .clk(clk), .rst_n(rst_n), .cfg_theta(8'd9), .cfg_mode_reset(1'b0), .cfg_period(17'd333),
        .cfg_clear_ref(1'b0), .cfg_clear_status(1'b0),
        .pix_valid(pix_valid), .pix_ready(pix_ready), .pix_data(pix_data), .pix_sop(pix_sop), .pix_eop(pix_eop),
        .ev_valid(ev_valid), .ev_ready(ev_ready), .ev_data(ev_data), .err_flags(err_flags), .ev_overflow_cnt(ev_overflow_cnt),
        .frame_cnt(frame_cnt), .ev_gen_cnt(ev_gen_cnt), .fifo_level(fifo_level), .idle(emu_idle));
    wire frame_done;
    frame_sync u_fs (.clk(clk), .rst_n(rst_n), .emu_idle(emu_idle), .frame_cnt(frame_cnt), .frame_done(frame_done));

    wire        clearing, in_q, core_busy, core_done, core_start, core_clear, overrun;
    wire [10:0] in_addr;
    wire [NC*28-1:0] score_flat, cur_flat, win_score;
    wire [19:0] spk1, spk2;
    wire [3:0]  pred;
    wire [15:0] windows;
    wire [7:0]  wcount;
    bin_builder u_b (.clk(clk), .rst_n(rst_n), .ev_valid(ev_valid), .ev_ready(ev_ready), .ev_data(ev_data),
                     .set_valid(1'b0), .set_addr(11'd0), .swap(frame_done), .clearing(clearing), .rd_addr(in_addr), .rd_q(in_q));
    snn_core #(.C1(`C1), .C2(`C2), .NCLS(NC), .W1FILE({`VEC, "/w1.hex"}), .W2FILE({`VEC, "/w2.hex"}), .WFFILE({`VEC, "/wfc.hex"})) u_c (
        .clk(clk), .rst_n(rst_n), .start(core_start), .clear(core_clear), .busy(core_busy), .done(core_done),
        .in_addr(in_addr), .in_q(in_q), .score_flat(score_flat), .cur_flat(cur_flat), .spk1(spk1), .spk2(spk2));
    snn_ctrl #(.T(`NSTEP), .NCLS(NC)) u_t (
        .clk(clk), .rst_n(rst_n), .step_trig(frame_done), .restart(1'b0), .clear_flags(1'b0),
        .core_start(core_start), .core_clear(core_clear), .core_busy(core_busy), .core_done(core_done), .score_flat(score_flat),
        .pred(pred), .win_score(win_score), .windows(windows), .wcount(wcount), .overrun(overrun));

    reg [31:0] exps   [0:`NSTEP*(NC+2)-1];
    reg [31:0] expf   [0:NC];
    initial begin
        $readmemh({`VEC, "/expect_steps.hex"}, exps);
        $readmemh({`VEC, "/expect_final.hex"}, expf);
    end

    integer k, errs, checked, cyc_start;
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
            checked = checked + 1;
            $display("info: step %0d spikes l1 %0d l2 %0d, %0d clocks", ck_step, spk1, spk2, $time / 40 - cyc_start);
            ck_step = ck_step + 1;
        end
    end

    initial begin
        errs = 0; checked = 0; cyc_start = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        wait (windows == 16'd1);                 // NSTEP frames processed
        repeat (10) @(posedge clk);
        if ({28'd0, pred} !== expf[0]) begin errs = errs + 1; $display("MISMATCH predicted class %0d expected %0d", pred, expf[0]); end
        for (k = 0; k < NC; k = k + 1) begin
            got = {{4{win_score[k*28+27]}}, win_score[k*28 +: 28]};
            if (got !== expf[1+k]) begin errs = errs + 1; $display("MISMATCH window score %0d got %0d exp %0d", k, $signed(got), $signed(expf[1+k])); end
        end
        if (overrun)   begin errs = errs + 1; $display("MISMATCH a step was skipped (overrun)"); end
        if (|err_flags) begin errs = errs + 1; $display("MISMATCH emulator error flags %b", err_flags); end
        $display("info: %0d steps checked, window class %0d, emulator events %0d, %0d errors", checked, pred, ev_gen_cnt, errs);
        if (errs == 0 && checked == `NSTEP) $display("RESULT: PASS");
        else $display("RESULT: FAIL");
        $finish;
    end

    initial begin
        #4000000000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end
endmodule
