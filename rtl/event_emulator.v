`timescale 1ns/1ps
// event_emulator.v -- frame-based DVS emulator. Bit-exact to model/emulator_int.py.
//
// Pixel stream in  : pix_valid/pix_ready, pix_data[7:0], pix_sop (first pixel of a frame),
//                    pix_eop (last pixel of a frame). Raster order, W*H pixels per frame.
//                    pix_ready is low while the event FIFO is nearly full (backpressure, no loss).
// Event stream out : ev_valid/ev_ready, ev_data[31:0] = {pol, y[6:0], x[6:0], ts[16:0]}.
//                    One event per pixel per frame at most, raster order.
//
// Pipeline (all registered):
//   stage 0 (pixel accepted): log-LUT read and reference-RAM read are issued, coordinates advance.
//   stage 1 (one clock later): d = log - ref and threshold compare, flags registered.
//   stage 2: reference update and event to FIFO.
// Reference RAM is read at stage 0 and written at stage 2 for the same address; the same pixel
// is read again one frame later, so there is no read-during-write hazard as long as W*H >= 8.
//
// cfg_* must be static while a frame is in flight. cfg_clear_ref (1-clock pulse, between frames)
// forgets the reference: the next frame is an init frame (no events) and timestamps restart at 0.
//
// Status (never silent): err_flags bit0 = sop at a wrong place / pixel outside a frame,
//   bit1 = eop too early, bit2 = last pixel without eop, bit3 = event FIFO overflow.
module event_emulator #(
    parameter W        = 128,
    parameter H        = 128,
    parameter FIFO_AW  = 10,
    parameter LUT_FILE = "rtl/log_lut.hex"
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [7:0]  cfg_theta,        // >= 1 (0 is treated as 1)
    input  wire        cfg_mode_reset,   // 0: step ref by theta, 1: reset ref to current
    input  wire [16:0] cfg_period,       // frame period in 100 us ticks
    input  wire        cfg_clear_ref,
    input  wire        cfg_clear_status,

    input  wire        pix_valid,
    output wire        pix_ready,
    input  wire [7:0]  pix_data,
    input  wire        pix_sop,
    input  wire        pix_eop,

    output wire        ev_valid,
    input  wire        ev_ready,
    output wire [31:0] ev_data,

    output reg  [3:0]  err_flags,
    output reg  [15:0] ev_overflow_cnt,
    output reg  [31:0] frame_cnt,
    output reg  [31:0] ev_gen_cnt,
    output wire [FIFO_AW:0] fifo_level,
    output wire        idle              // nothing in flight: no open frame, pipeline and FIFO empty
);
    localparam NPIX = W * H;
    localparam AW   = $clog2(NPIX);
    localparam [AW-1:0] LAST_ADDR = NPIX - 1;
    localparam [6:0]    LAST_X    = W - 1;
    localparam [6:0]    LAST_Y    = H - 1;

    // ---------------- stage 0: accept pixel, track position ----------------
    reg           in_frame;
    reg           have_ref;
    reg           init_cur;
    reg           ts_started;
    reg [16:0]    ts_cur;
    reg [AW-1:0]  a_cnt;
    reg [6:0]     x_cnt, y_cnt;

    wire fire = pix_valid && pix_ready;
    assign pix_ready = rst_n && (fifo_level < ((1 << FIFO_AW) - 8));

    wire          resync   = pix_sop && in_frame;                     // sop while a frame was open
    wire [AW-1:0] a_use    = pix_sop ? {AW{1'b0}} : a_cnt;
    wire [6:0]    x_use    = pix_sop ? 7'd0 : x_cnt;
    wire [6:0]    y_use    = pix_sop ? 7'd0 : y_cnt;
    wire          valid_px = fire && (pix_sop || in_frame);           // pixel belongs to a frame
    wire [16:0]   ts_use   = pix_sop ? (ts_started ? ts_cur + cfg_period : 17'd0) : ts_cur;
    wire          init_use = pix_sop ? !have_ref : init_cur;
    wire          last_px  = (a_use == LAST_ADDR);

    wire [7:0] lut_q;
    wire [7:0] ref_q;

    // stage 1 registers
    reg           s1_valid, s1_init;
    reg [AW-1:0]  s1_addr;
    reg [6:0]     s1_x, s1_y;
    reg [16:0]    s1_ts;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_frame <= 1'b0; have_ref <= 1'b0; init_cur <= 1'b0; ts_started <= 1'b0;
            ts_cur <= 17'd0; a_cnt <= 0; x_cnt <= 7'd0; y_cnt <= 7'd0;
            s1_valid <= 1'b0; s1_init <= 1'b0; s1_addr <= 0; s1_x <= 7'd0; s1_y <= 7'd0; s1_ts <= 17'd0;
        end else begin
            s1_valid <= valid_px;
            s1_init  <= init_use;
            s1_addr  <= a_use;
            s1_x     <= x_use;
            s1_y     <= y_use;
            s1_ts    <= ts_use;

            if (cfg_clear_ref) begin
                have_ref <= 1'b0; ts_started <= 1'b0;
            end
            if (valid_px) begin
                in_frame <= !last_px && !pix_eop;
                if (pix_sop) begin
                    init_cur   <= init_use;
                    ts_cur     <= ts_use;
                    ts_started <= 1'b1;
                end
                if (last_px || pix_eop) have_ref <= 1'b1;
                a_cnt <= last_px ? {AW{1'b0}} : a_use + 1'b1;
                if (x_use == LAST_X) begin
                    x_cnt <= 7'd0;
                    y_cnt <= (y_use == LAST_Y) ? 7'd0 : y_use + 7'd1;
                end else begin
                    x_cnt <= x_use + 7'd1;
                    y_cnt <= y_use;
                end
            end
        end
    end

    log_lut #(.INIT_FILE(LUT_FILE)) u_lut (.clk(clk), .addr(pix_data), .q(lut_q));

    // ---------------- stage 1: compare (registered flags) ----------------
    wire [7:0]  theta = (cfg_theta == 8'd0) ? 8'd1 : cfg_theta;
    wire signed [9:0] d  = $signed({2'b00, lut_q}) - $signed({2'b00, ref_q});
    wire signed [9:0] th = $signed({2'b00, theta});
    wire        on_ev  = s1_valid && !s1_init && (d >=  th);
    wire        off_ev = s1_valid && !s1_init && (d <= -th);

    reg           s2_valid, s2_init, s2_on, s2_off;
    reg [7:0]     s2_cur, s2_ref, s2_theta;
    reg [AW-1:0]  s2_addr;
    reg [6:0]     s2_x, s2_y;
    reg [16:0]    s2_ts;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0; s2_init <= 1'b0; s2_on <= 1'b0; s2_off <= 1'b0;
            s2_cur <= 8'd0; s2_ref <= 8'd0; s2_theta <= 8'd1; s2_addr <= 0; s2_x <= 7'd0; s2_y <= 7'd0; s2_ts <= 17'd0;
        end else begin
            s2_valid <= s1_valid; s2_init <= s1_init; s2_on <= on_ev; s2_off <= off_ev;
            s2_cur <= lut_q; s2_ref <= ref_q; s2_theta <= theta;
            s2_addr <= s1_addr; s2_x <= s1_x; s2_y <= s1_y; s2_ts <= s1_ts;
        end
    end

    // ---------------- stage 2: update reference, emit event ----------------
    wire        ev_new   = s2_on || s2_off;
    wire [7:0]  ref_step = s2_on ? (s2_ref + s2_theta) : (s2_ref - s2_theta);
    wire [7:0]  ref_new  = s2_init ? s2_cur : (cfg_mode_reset ? s2_cur : ref_step);
    wire        ref_we   = s2_valid && (s2_init || ev_new);

    sdp_ram #(.DW(8), .AW(AW)) u_ref (
        .clk(clk),
        .we(ref_we),  .waddr(s2_addr), .wdata(ref_new),
        .re(1'b1),    .raddr(a_use),   .rdata(ref_q)
    );

    // ---------------- event FIFO ----------------
    wire fifo_overflow;
    event_fifo #(.AW(FIFO_AW), .DW(32)) u_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_en(ev_new), .wr_data({s2_on, s2_y, s2_x, s2_ts}),
        .level(fifo_level), .overflow(fifo_overflow),
        .rd_data(ev_data), .rd_valid(ev_valid), .rd_ready(ev_ready)
    );

    assign idle = !in_frame && !s1_valid && !s2_valid && (fifo_level == 0) && !ev_valid;

    // ---------------- status ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            err_flags <= 4'd0; ev_overflow_cnt <= 16'd0; frame_cnt <= 32'd0; ev_gen_cnt <= 32'd0;
        end else if (cfg_clear_status) begin
            err_flags <= 4'd0; ev_overflow_cnt <= 16'd0; frame_cnt <= 32'd0; ev_gen_cnt <= 32'd0;
        end else begin
            if (fire && !pix_sop && !in_frame)          err_flags[0] <= 1'b1;   // stray pixel (dropped)
            if (resync)                                 err_flags[0] <= 1'b1;   // sop inside an open frame
            if (valid_px && pix_eop && !last_px)        err_flags[1] <= 1'b1;
            if (valid_px && last_px && !pix_eop)        err_flags[2] <= 1'b1;
            if (fifo_overflow) begin
                err_flags[3] <= 1'b1;
                if (ev_overflow_cnt != 16'hFFFF) ev_overflow_cnt <= ev_overflow_cnt + 16'd1;
            end
            if (valid_px && (pix_eop || last_px))       frame_cnt <= frame_cnt + 32'd1;
            if (ev_new)                                 ev_gen_cnt <= ev_gen_cnt + 32'd1;
        end
    end
endmodule
