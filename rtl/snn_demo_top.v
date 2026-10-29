`timescale 1ns/1ps
// snn_demo_top.v -- on-chip demo of the whole chain (roadmap Phase 5/6): frames -> event emulator -> 2x32x32 bins -> integer SNN
// -> class of the last decision window. No HPS. Single 25 MHz clock domain (PLL from CLOCK_50).
//
// Two sources, SW[9]:
//   SW[9] = 0  REPLAY (default): clips of real KTH video, already turned into event bins by the Python emulator model, are
//              played into the SNN at 30 steps/s (rtl/clip_replay.v). KEY[3] selects the next clip. Tests the SNN on hardware.
//   SW[9] = 1  LIVE: the on-chip pattern source (a moving square) -> RTL event emulator -> bin builder -> SNN, and the VGA event
//              view as in demo_top. Tests the whole chain on hardware; the pattern is not an action, its class is arbitrary
//              but must equal the model's (tb/snn_chain_tb.v).
//
// Controls
//   KEY[0] reset      KEY[1] forget the emulator reference    KEY[2] clear error flags / counters    KEY[3] next clip (restarts the window)
//   SW[4:0] emulator threshold theta = 2*SW+1 (4 -> 9, the value the network was trained with, LIVE only)
//   SW[5] emulator reference update 0 = step, 1 = reset (LIVE only)    SW[7:6] VGA event window    SW[8] pause the source
// Indicators
//   HEX0 : class of the last finished window (F = none yet)         HEX1 : clip number (REPLAY) / F (LIVE)
//   HEX2 : the clip's true class (REPLAY) / F (LIVE)                 HEX3 : windows finished (low nibble)
//   HEX5,HEX4 : steps done in the current window (hex, 0 .. T-1)
//   LEDR[0] heartbeat, [1] PLL locked, [2] SNN core busy, [3] any error (emulator flags or a missed step), [9:4] one-hot class of the last window
module snn_demo_top #(
    parameter FRAME_CLKS = 833333,
    parameter LUT_FILE   = "rtl/log_lut.hex",
    parameter C1         = 16,
    parameter C2         = 32,
    parameter NCLS       = 6,
    parameter T          = 32,
    parameter NCLIP      = 6,
    parameter CLIP_STEPS = 64,
    parameter W1FILE     = "rtl/weights/kth6/w1.hex",
    parameter W2FILE     = "rtl/weights/kth6/w2.hex",
    parameter WFFILE     = "rtl/weights/kth6/wfc.hex",
    parameter CLIPFILE   = "rtl/weights/kth6/clips.hex",
    parameter LBLFILE    = "rtl/weights/kth6/labels.hex"
) (
    input  wire        CLOCK_50,
    input  wire [3:0]  KEY,
    input  wire [9:0]  SW,
    output wire [9:0]  LEDR,
    output wire [6:0]  HEX0, HEX1, HEX2, HEX3, HEX4, HEX5,
    output wire        VGA_CLK,
    output wire        VGA_HS,
    output wire        VGA_VS,
    output wire [7:0]  VGA_R,
    output wire [7:0]  VGA_G,
    output wire [7:0]  VGA_B,
    output wire        VGA_BLANK_N,
    output wire        VGA_SYNC_N
);
    // ---------------- clock, reset, keys ----------------
    wire clk25, pll_locked, rst_n;
    pll25 u_pll (.refclk(CLOCK_50), .rst(1'b0), .outclk_0(clk25), .locked(pll_locked));
    reset_sync #(.STAGES(3)) u_rst (.clk(clk25), .rst_n_in(KEY[0] & pll_locked), .rst_n_out(rst_n));

    reg [2:0] k1, k2, k3;
    reg [1:0] s9;
    always @(posedge clk25) begin
        k1 <= {k1[1:0], KEY[1]}; k2 <= {k2[1:0], KEY[2]}; k3 <= {k3[1:0], KEY[3]}; s9 <= {s9[0], SW[9]};
    end
    wire clear_ref    = k1[2] & ~k1[1];
    wire clear_status = k2[2] & ~k2[1];
    wire next_clip    = k3[2] & ~k3[1];
    wire live         = s9[1];
    wire live_change  = s9[1] ^ s9[0];

    reg [2:0] clip_sel;
    always @(posedge clk25 or negedge rst_n) begin
        if (!rst_n) clip_sel <= 3'd0;
        else if (next_clip) clip_sel <= (clip_sel == NCLIP - 1) ? 3'd0 : clip_sel + 3'd1;
    end
    wire restart = next_clip | live_change;

    // ---------------- VGA timing ----------------
    wire        hs_n, vs_n, de, sof, vblank;
    wire [10:0] vx, vy;
    vga_timing u_timing (.clk(clk25), .rst_n(rst_n), .hs_n(hs_n), .vs_n(vs_n), .de(de), .sof(sof),
                         .vblank(vblank), .x(vx), .y(vy));

    // ---------------- LIVE: pattern source -> emulator ----------------
    wire        pix_valid, pix_ready, pix_sop, pix_eop;
    wire [7:0]  pix_data;
    wire [15:0] src_frames;
    pix_source #(.FRAME_CLKS(FRAME_CLKS)) u_src (
        .clk(clk25), .rst_n(rst_n), .run(live & ~SW[8]),
        .pix_valid(pix_valid), .pix_ready(pix_ready), .pix_data(pix_data),
        .pix_sop(pix_sop), .pix_eop(pix_eop), .frame_idx(src_frames));

    wire        ev_valid, ev_ready, emu_idle;
    wire [31:0] ev_data, frame_cnt, ev_gen_cnt;
    wire [3:0]  err_flags;
    wire [15:0] ev_overflow_cnt;
    wire [10:0] fifo_level;
    event_emulator #(.W(128), .H(128), .FIFO_AW(10), .LUT_FILE(LUT_FILE)) u_emu (
        .clk(clk25), .rst_n(rst_n),
        .cfg_theta({2'b00, SW[4:0], 1'b0} + 8'd1), .cfg_mode_reset(SW[5]), .cfg_period(17'd333),
        .cfg_clear_ref(clear_ref), .cfg_clear_status(clear_status),
        .pix_valid(pix_valid), .pix_ready(pix_ready), .pix_data(pix_data),
        .pix_sop(pix_sop), .pix_eop(pix_eop),
        .ev_valid(ev_valid), .ev_ready(ev_ready), .ev_data(ev_data),
        .err_flags(err_flags), .ev_overflow_cnt(ev_overflow_cnt), .frame_cnt(frame_cnt),
        .ev_gen_cnt(ev_gen_cnt), .fifo_level(fifo_level), .idle(emu_idle));

    wire frame_done;
    frame_sync u_fs (.clk(clk25), .rst_n(rst_n), .emu_idle(emu_idle), .frame_cnt(frame_cnt), .frame_done(frame_done));

    // ---------------- REPLAY: clip player ----------------
    reg [19:0] ftimer;
    wire       tick = (ftimer == FRAME_CLKS - 1);
    always @(posedge clk25 or negedge rst_n) begin
        if (!rst_n) ftimer <= 20'd0;
        else if (tick) ftimer <= 20'd0;
        else if (!SW[8]) ftimer <= ftimer + 20'd1;
    end
    wire        rp_set, rp_swap, b_clearing;
    wire [10:0] rp_addr;
    wire [7:0]  rp_step;
    clip_replay #(.NCLIP(NCLIP), .NSTEP(CLIP_STEPS), .FILE(CLIPFILE)) u_rp (
        .clk(clk25), .rst_n(rst_n), .tick(tick & ~live), .restart(restart), .clip(clip_sel), .clearing(b_clearing),
        .set_valid(rp_set), .set_addr(rp_addr), .swap(rp_swap), .step_idx(rp_step));

    // ---------------- bin builder + SNN ----------------
    wire        swap = live ? frame_done : rp_swap;
    wire        v_ready, b_ready;
    wire [10:0] in_addr;
    wire        in_q;
    bin_builder u_bb (
        .clk(clk25), .rst_n(rst_n),
        .ev_valid(ev_valid & v_ready & live), .ev_ready(b_ready), .ev_data(ev_data),
        .set_valid(rp_set), .set_addr(rp_addr), .swap(swap), .clearing(b_clearing),
        .rd_addr(in_addr), .rd_q(in_q));

    wire        core_start, core_clear, core_busy, core_done, overrun;
    wire [NCLS*28-1:0] score_flat, cur_flat, win_score;
    wire [19:0] spk1, spk2;
    snn_core #(.C1(C1), .C2(C2), .NCLS(NCLS), .W1FILE(W1FILE), .W2FILE(W2FILE), .WFFILE(WFFILE)) u_core (
        .clk(clk25), .rst_n(rst_n), .start(core_start), .clear(core_clear), .busy(core_busy), .done(core_done),
        .in_addr(in_addr), .in_q(in_q), .score_flat(score_flat), .cur_flat(cur_flat), .spk1(spk1), .spk2(spk2));

    wire [3:0]  pred;
    wire [15:0] windows;
    wire [7:0]  wcount;
    snn_ctrl #(.T(T), .NCLS(NCLS)) u_ctl (
        .clk(clk25), .rst_n(rst_n), .step_trig(swap), .restart(restart), .clear_flags(clear_status),
        .core_start(core_start), .core_clear(core_clear), .core_busy(core_busy), .core_done(core_done), .score_flat(score_flat),
        .pred(pred), .win_score(win_score), .windows(windows), .wcount(wcount), .overrun(overrun));

    // ---------------- event view -> VGA (LIVE) ----------------
    wire [7:0]  r, g, b;
    wire        de_o, hs_o, vs_o;
    wire [15:0] disp_first, disp_last;
    event_view u_view (
        .clk(clk25), .rst_n(rst_n),
        .ev_valid(ev_valid & b_ready), .ev_ready(v_ready), .ev_data(ev_data),
        .emu_idle(emu_idle), .frame_cnt(frame_cnt), .cfg_win(SW[7:6]),
        .x(vx), .y(vy), .de(de), .hs_n(hs_n), .vs_n(vs_n), .vblank(vblank),
        .r(r), .g(g), .b(b), .de_o(de_o), .hs_o(hs_o), .vs_o(vs_o),
        .disp_first(disp_first), .disp_last(disp_last));
    assign ev_ready = v_ready & b_ready;

    assign VGA_CLK     = clk25;
    assign VGA_HS      = hs_o;
    assign VGA_VS      = vs_o;
    assign VGA_R       = r;
    assign VGA_G       = g;
    assign VGA_B       = b;
    assign VGA_BLANK_N = de_o;
    assign VGA_SYNC_N  = 1'b0;

    // ---------------- status ----------------
    reg [3:0] lbl [0:7];
    integer li;
    initial begin
        for (li = 0; li < 8; li = li + 1) lbl[li] = 4'hF;
        $readmemh(LBLFILE, lbl);
    end
    wire [3:0] pred_show = (windows == 16'd0) ? 4'hF : pred;
    reg [25:0] hb50 = 26'd0;
    always @(posedge CLOCK_50) hb50 <= hb50 + 26'd1;
    assign LEDR[0] = hb50[25];
    assign LEDR[1] = pll_locked;
    assign LEDR[2] = core_busy;
    assign LEDR[3] = (|err_flags) | overrun;
    genvar gi;
    generate
        for (gi = 0; gi < 6; gi = gi + 1) begin : g_led
            assign LEDR[4 + gi] = (windows != 16'd0) && (pred == gi);
        end
    endgenerate

    hex7seg h0 (.d(pred_show),                       .seg_n(HEX0));
    hex7seg h1 (.d(live ? 4'hF : {1'b0, clip_sel}),  .seg_n(HEX1));
    hex7seg h2 (.d(live ? 4'hF : lbl[clip_sel]),     .seg_n(HEX2));
    hex7seg h3 (.d(windows[3:0]),                    .seg_n(HEX3));
    hex7seg h4 (.d(wcount[3:0]),                     .seg_n(HEX4));
    hex7seg h5 (.d(wcount[7:4]),                     .seg_n(HEX5));
endmodule
