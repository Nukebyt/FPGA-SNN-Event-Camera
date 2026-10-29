`timescale 1ns/1ps
// demo_top.v -- on-chip demo (roadmap 1.10 / 1.11 / 4.11): pattern source -> event emulator -> VGA event view.
// No HPS involved. Single 25 MHz clock domain (PLL from CLOCK_50); everything is lockstep with the VGA pixel clock.
//
// Controls
//   KEY[0]   : reset (press = reset)           KEY[1] : forget the reference (next frame is an init frame)
//   KEY[2]   : clear status flags and counters
//   SW[4:0]  : contrast threshold theta = 2*SW + 1 LUT LSBs (SW = 4 -> theta 9 ~ 0.2 ln units)
//   SW[5]    : 0 = reference moves by theta ("step"), 1 = reference reset to current ("reset")
//   SW[7:6]  : event view window 1, 2, 4, 8 emulator frames
//   SW[8]    : 1 = pause the frame source
// Indicators
//   LEDR[0] clk50 heartbeat, [1] PLL locked, [2] out of reset, [3] emulator idle, [4] any error flag,
//   [5] source frame in progress, [9:6] SW[9:6] mirror
//   HEX3..HEX0 : emulator frame counter (hex), HEX4 : err_flags, HEX5 : event overflow count (low nibble)
module demo_top #(
    parameter FRAME_CLKS = 833333,          // source frame period in 25 MHz clocks (30 fps)
    parameter LUT_FILE   = "rtl/log_lut.hex"
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
    // ---------------- clock and reset ----------------
    wire clk25, pll_locked, rst_n;
    pll25 u_pll (.refclk(CLOCK_50), .rst(1'b0), .outclk_0(clk25), .locked(pll_locked));
    reset_sync #(.STAGES(3)) u_rst (.clk(clk25), .rst_n_in(KEY[0] & pll_locked), .rst_n_out(rst_n));

    // KEY[1], KEY[2]: synchronise, one-clock pulse on press
    reg [2:0] k1, k2;
    always @(posedge clk25) begin k1 <= {k1[1:0], KEY[1]}; k2 <= {k2[1:0], KEY[2]}; end
    wire clear_ref    = k1[2] & ~k1[1];
    wire clear_status = k2[2] & ~k2[1];

    // ---------------- VGA timing ----------------
    wire        hs_n, vs_n, de, sof, vblank;
    wire [10:0] vx, vy;
    vga_timing u_timing (.clk(clk25), .rst_n(rst_n), .hs_n(hs_n), .vs_n(vs_n), .de(de), .sof(sof),
                         .vblank(vblank), .x(vx), .y(vy));

    // ---------------- source -> emulator ----------------
    wire        pix_valid, pix_ready, pix_sop, pix_eop;
    wire [7:0]  pix_data;
    wire [15:0] src_frames;
    pix_source #(.FRAME_CLKS(FRAME_CLKS)) u_src (
        .clk(clk25), .rst_n(rst_n), .run(~SW[8]),
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

    // ---------------- event view -> VGA ----------------
    wire [7:0] r, g, b;
    wire       de_o, hs_o, vs_o;
    wire [15:0] disp_first, disp_last;
    event_view u_view (
        .clk(clk25), .rst_n(rst_n),
        .ev_valid(ev_valid), .ev_ready(ev_ready), .ev_data(ev_data),
        .emu_idle(emu_idle), .frame_cnt(frame_cnt), .cfg_win(SW[7:6]),
        .x(vx), .y(vy), .de(de), .hs_n(hs_n), .vs_n(vs_n), .vblank(vblank),
        .r(r), .g(g), .b(b), .de_o(de_o), .hs_o(hs_o), .vs_o(vs_o),
        .disp_first(disp_first), .disp_last(disp_last));

    assign VGA_CLK     = clk25;
    assign VGA_HS      = hs_o;
    assign VGA_VS      = vs_o;
    assign VGA_R       = r;
    assign VGA_G       = g;
    assign VGA_B       = b;
    assign VGA_BLANK_N = de_o;
    assign VGA_SYNC_N  = 1'b0;

    // ---------------- status ----------------
    reg [25:0] hb50 = 26'd0;
    always @(posedge CLOCK_50) hb50 <= hb50 + 26'd1;
    assign LEDR[0]   = hb50[25];
    assign LEDR[1]   = pll_locked;
    assign LEDR[2]   = rst_n;
    assign LEDR[3]   = emu_idle;
    assign LEDR[4]   = |err_flags;
    assign LEDR[5]   = pix_valid;
    assign LEDR[9:6] = SW[9:6];

    hex7seg h0 (.d(frame_cnt[3:0]),        .seg_n(HEX0));
    hex7seg h1 (.d(frame_cnt[7:4]),        .seg_n(HEX1));
    hex7seg h2 (.d(frame_cnt[11:8]),       .seg_n(HEX2));
    hex7seg h3 (.d(frame_cnt[15:12]),      .seg_n(HEX3));
    hex7seg h4 (.d(err_flags),             .seg_n(HEX4));
    hex7seg h5 (.d(ev_overflow_cnt[3:0]),  .seg_n(HEX5));
endmodule
