`timescale 1ns/1ps
// snn_engine.v -- the event-camera + SNN engine behind the HPS (same interface pattern as the night-vision project's nv_engine):
// CSR slave on the lightweight bridge, one Avalon-MM read master on the HPS F2H SDRAM port, one 25 MHz clock end to end.
//
//   HPS DDR3 input ring --dma_rd--> fifo_64to8 --> event_emulator --> bin_builder --> snn_core --> CSR (class, scores, counters)
//
// One frame = 128 x 128 bytes (8-bit luma, raster order) in slot (IN_WR_SEQ % 4) of the input ring (slot stride 0x4000 bytes).
// Software writes a frame, then increments IN_WR_SEQ. When enabled and idle, the engine reads the next frame through the DMA,
// streams its pixels into the event emulator (back-pressure stalls the byte FIFO, nothing is lost), waits until the emulator is
// idle (frame_sync), hands the 2x32x32 event bins to the SNN (one step, 205,832 clocks) and publishes the result. A new frame is
// only started after the previous step finished, so the SNN never overruns; a frame takes about 0.7 ms (pixels) + 8.2 ms (SNN).
// One step = one frame regardless of the wall-clock rate at which software delivers frames (the emulator timestamps frames
// by count, as in training). The decision window (T steps) and its class are produced by snn_ctrl exactly as in snn_demo_top.
//
// CSR map (word addresses, 32-bit, read latency 1): see docs/snn_soc.md. Every result register is a snapshot taken when
// DONE_SEQ increments, so software that sees DONE_SEQ change reads a consistent set.
module snn_engine #(
    parameter W        = 128,
    parameter H        = 128,
    parameter C1       = 16,
    parameter C2       = 32,
    parameter NCLS     = 6,
    parameter T        = 32,
    parameter [31:0] BUILD_ID = 32'h534E0001,
    parameter [31:0] CLOCK_HZ = 32'd25000000,
    parameter LUT_FILE = "rtl/log_lut.hex",
    parameter W1FILE   = "rtl/weights/kth6/w1.hex",
    parameter W2FILE   = "rtl/weights/kth6/w2.hex",
    parameter WFFILE   = "rtl/weights/kth6/wfc.hex"
) (
    input  wire        clk,
    input  wire        rst_n,
    // CSR slave (lightweight bridge)
    input  wire [7:0]  csr_address,
    input  wire        csr_read,
    input  wire        csr_write,
    input  wire [31:0] csr_writedata,
    output reg  [31:0] csr_readdata,
    // input DMA master (64-bit, byte addresses) -> F2H SDRAM port
    output wire [31:0] rd_address,
    output wire        rd_read,
    input  wire        rd_waitrequest,
    input  wire [63:0] rd_readdata,
    input  wire        rd_readdatavalid,
    // status for LEDs / HEX
    output wire [3:0]  st_pred,
    output wire [7:0]  st_wcount,
    output wire        st_ready,
    output wire        st_enabled,
    output wire        st_error,
    output wire [31:0] st_done
);
    localparam NPIX   = W * H;
    localparam NWORDS = NPIX / 8;

    // ------------------------------------------------------------------ CSR registers
    reg         enable, mode_reset, req_reset, req_clear_ref, clr_sticky;
    reg [31:0]  in_base, in_wr_seq, scratch;
    reg [7:0]   theta;

    // soft reset: everything behind the CSR block is held in reset for 8 clocks when software writes CONTROL[1]
    reg [3:0]   soft_cnt;
    reg         core_rst_n;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin soft_cnt <= 4'd0; core_rst_n <= 1'b0; end
        else begin
            if (req_reset) soft_cnt <= 4'd8;
            else if (soft_cnt != 4'd0) soft_cnt <= soft_cnt - 4'd1;
            core_rst_n <= (soft_cnt == 4'd0) && !req_reset;
        end
    end

    // ------------------------------------------------------------------ scheduler and input path
    reg  [31:0] in_rd_seq, done_seq;
    reg         f_active, await_done, pre_start, dma_start, f_flush;
    reg  [31:0] dma_base;
    reg  [13:0] pcount;
    reg  [31:0] lat_cnt, lat_last, ev_start;

    wire [9:0]  f_lvl;
    wire [7:0]  f_byte;
    wire        f_empty, f_wr_en, rd_busy, rd_done;
    wire [63:0] f_wr_data;
    wire        pix_valid = f_active && !f_empty;
    wire        pix_ready;
    wire        pix_fire  = pix_valid && pix_ready;
    wire        pix_sop   = pix_valid && (pcount == 14'd0);
    wire        pix_eop   = pix_valid && (pcount == NPIX - 1);

    fifo_64to8 #(.AW(8)) u_fifo (.clk(clk), .rst_n(core_rst_n), .flush(f_flush), .wr_en(f_wr_en), .wr_data(f_wr_data), .lvl(f_lvl),
        .rd_en(pix_fire), .rd_data(f_byte), .empty(f_empty));
    dma_rd #(.NWORDS(NWORDS), .AW(8), .LIMIT(252)) u_rd (.clk(clk), .rst_n(core_rst_n), .start(dma_start), .base(dma_base), .lvl(f_lvl),
        .busy(rd_busy), .done(rd_done), .fifo_wr_en(f_wr_en), .fifo_wr_data(f_wr_data),
        .m_address(rd_address), .m_read(rd_read), .m_waitrequest(rd_waitrequest), .m_readdata(rd_readdata),
        .m_readdatavalid(rd_readdatavalid));

    // ------------------------------------------------------------------ emulator -> bins -> SNN
    wire        ev_valid, ev_ready, emu_idle;
    wire [31:0] ev_data, frame_cnt, ev_gen_cnt;
    wire [3:0]  err_flags;
    wire [15:0] ev_overflow_cnt;
    wire [10:0] fifo_level;
    event_emulator #(.W(W), .H(H), .FIFO_AW(10), .LUT_FILE(LUT_FILE)) u_emu (
        .clk(clk), .rst_n(core_rst_n),
        .cfg_theta(theta), .cfg_mode_reset(mode_reset), .cfg_period(17'd333),
        .cfg_clear_ref(req_clear_ref), .cfg_clear_status(clr_sticky),
        .pix_valid(pix_valid), .pix_ready(pix_ready), .pix_data(f_byte), .pix_sop(pix_sop), .pix_eop(pix_eop),
        .ev_valid(ev_valid), .ev_ready(ev_ready), .ev_data(ev_data),
        .err_flags(err_flags), .ev_overflow_cnt(ev_overflow_cnt), .frame_cnt(frame_cnt), .ev_gen_cnt(ev_gen_cnt),
        .fifo_level(fifo_level), .idle(emu_idle));

    wire frame_done;
    frame_sync u_fs (.clk(clk), .rst_n(core_rst_n), .emu_idle(emu_idle), .frame_cnt(frame_cnt), .frame_done(frame_done));

    wire        b_clearing, in_q;
    wire [10:0] in_addr;
    bin_builder u_bb (.clk(clk), .rst_n(core_rst_n), .ev_valid(ev_valid), .ev_ready(ev_ready), .ev_data(ev_data),
        .set_valid(1'b0), .set_addr(11'd0), .swap(frame_done), .clearing(b_clearing), .rd_addr(in_addr), .rd_q(in_q));

    wire        core_start, core_clear, core_busy, core_done;
    wire [NCLS*28-1:0] score_flat, cur_flat, win_score;
    wire [19:0] spk1, spk2;
    snn_core #(.C1(C1), .C2(C2), .NCLS(NCLS), .W1FILE(W1FILE), .W2FILE(W2FILE), .WFFILE(WFFILE)) u_core (
        .clk(clk), .rst_n(core_rst_n), .start(core_start), .clear(core_clear), .busy(core_busy), .done(core_done),
        .in_addr(in_addr), .in_q(in_q), .score_flat(score_flat), .cur_flat(cur_flat), .spk1(spk1), .spk2(spk2));

    wire [3:0]  pred;
    wire [15:0] windows;
    wire [7:0]  wcount;
    wire        overrun, ctrl_ready, step_done;
    snn_ctrl #(.T(T), .NCLS(NCLS)) u_ctl (
        .clk(clk), .rst_n(core_rst_n), .step_trig(frame_done), .restart(1'b0), .clear_flags(clr_sticky),
        .core_start(core_start), .core_clear(core_clear), .core_busy(core_busy), .core_done(core_done), .score_flat(score_flat),
        .pred(pred), .win_score(win_score), .windows(windows), .wcount(wcount), .overrun(overrun),
        .ready(ctrl_ready), .step_done(step_done));

    // ------------------------------------------------------------------ sequencing
    wire        b_idle    = !b_clearing;
    wire        seq_ready = (in_wr_seq != in_rd_seq);
    wire        can_start = enable && ctrl_ready && b_idle && !f_active && !pre_start && !await_done && seq_ready;
    reg         step_done_d;
    reg [31:0]  step_cnt, step_last;

    // result snapshots (taken one clock after step_done, when snn_ctrl has updated pred / windows / wcount)
    reg [3:0]   s_pred;
    reg [15:0]  s_windows;
    reg [7:0]   s_wcount;
    reg [19:0]  s_spk1, s_spk2;
    reg [31:0]  s_events;
    reg signed [27:0] s_cur [0:NCLS-1];
    reg signed [27:0] s_score [0:NCLS-1];
    reg signed [27:0] s_win [0:NCLS-1];
    reg [NCLS*28-1:0] cur_cap, score_cap;
    reg [19:0]  spk1_cap, spk2_cap;
    integer k;
    reg st_overrun, st_emu_err;

    always @(posedge clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            f_active <= 1'b0; await_done <= 1'b0; pre_start <= 1'b0; dma_start <= 1'b0; f_flush <= 1'b0; dma_base <= 32'd0;
            pcount <= 14'd0; in_rd_seq <= 32'd0; done_seq <= 32'd0; lat_cnt <= 32'd0; lat_last <= 32'd0; ev_start <= 32'd0;
            step_done_d <= 1'b0; step_cnt <= 32'd0; step_last <= 32'd0;
            s_pred <= 4'd0; s_windows <= 16'd0; s_wcount <= 8'd0; s_spk1 <= 20'd0; s_spk2 <= 20'd0; s_events <= 32'd0;
            cur_cap <= 0; score_cap <= 0; spk1_cap <= 20'd0; spk2_cap <= 20'd0; st_overrun <= 1'b0; st_emu_err <= 1'b0;
            for (k = 0; k < NCLS; k = k + 1) begin s_cur[k] <= 28'sd0; s_score[k] <= 28'sd0; s_win[k] <= 28'sd0; end
        end else begin
            dma_start <= 1'b0;
            f_flush   <= 1'b0;
            step_done_d <= step_done;
            if (clr_sticky) begin st_overrun <= 1'b0; st_emu_err <= 1'b0; end
            if (overrun)    st_overrun <= 1'b1;
            if (|err_flags) st_emu_err <= 1'b1;

            // start of a frame: flush the byte FIFO, start the DMA one clock later
            if (can_start) begin
                f_flush <= 1'b1; pre_start <= 1'b1; f_active <= 1'b1; await_done <= 1'b1; pcount <= 14'd0;
                dma_base <= in_base + {in_rd_seq[1:0], 14'd0};
                lat_cnt <= 32'd1; ev_start <= ev_gen_cnt;
            end else if (await_done) lat_cnt <= lat_cnt + 32'd1;
            if (pre_start && !can_start) begin dma_start <= 1'b1; pre_start <= 1'b0; end

            // pixels
            if (pix_fire) begin
                pcount <= pcount + 14'd1;
                if (pcount == NPIX - 1) begin f_active <= 1'b0; in_rd_seq <= in_rd_seq + 32'd1; end
            end

            // step timing
            if (core_start) step_cnt <= 32'd1;
            else if (core_busy) step_cnt <= step_cnt + 32'd1;

            // a step finished: capture what the core holds this very clock
            if (step_done) begin
                cur_cap <= cur_flat; score_cap <= score_flat; spk1_cap <= spk1; spk2_cap <= spk2;
                step_last <= step_cnt + 32'd1;
            end
            // one clock later: ctrl has updated pred / windows / wcount; publish a consistent set
            if (step_done_d) begin
                s_pred <= pred; s_windows <= windows; s_wcount <= wcount; s_spk1 <= spk1_cap; s_spk2 <= spk2_cap;
                for (k = 0; k < NCLS; k = k + 1) begin
                    s_cur[k]   <= cur_cap[k*28 +: 28];
                    s_score[k] <= score_cap[k*28 +: 28];
                    s_win[k]   <= win_score[k*28 +: 28];
                end
                s_events <= ev_gen_cnt - ev_start;
                lat_last <= lat_cnt;
                await_done <= 1'b0;
                done_seq <= done_seq + 32'd1;
            end
        end
    end

    // ------------------------------------------------------------------ CSR block (registered read, fixed latency 1)
    wire [31:0] status = {20'd0, err_flags, 3'd0, st_emu_err, st_overrun, enable, (f_active | await_done | core_busy), ctrl_ready};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            enable <= 1'b0; mode_reset <= 1'b0; req_reset <= 1'b0; req_clear_ref <= 1'b0; clr_sticky <= 1'b0;
            in_base <= 32'd0; in_wr_seq <= 32'd0; scratch <= 32'd0; theta <= 8'd9; csr_readdata <= 32'd0;
        end else begin
            req_reset <= 1'b0; req_clear_ref <= 1'b0; clr_sticky <= 1'b0;
            if (csr_write) begin
                case (csr_address)
                    8'd1: begin
                        enable <= csr_writedata[0];
                        mode_reset <= csr_writedata[3];
                        if (csr_writedata[1]) begin req_reset <= 1'b1; in_wr_seq <= 32'd0; end
                        if (csr_writedata[2]) req_clear_ref <= 1'b1;
                    end
                    8'd3: clr_sticky <= 1'b1;
                    8'd4: in_base <= csr_writedata;
                    8'd6: in_wr_seq <= csr_writedata;
                    8'd9: theta <= csr_writedata[7:0];
                    8'd11: scratch <= csr_writedata;
                    default: ;
                endcase
            end
            case (csr_address)
                8'd0:  csr_readdata <= BUILD_ID;
                8'd1:  csr_readdata <= {28'd0, mode_reset, 2'b00, enable};
                8'd2:  csr_readdata <= status;
                8'd4:  csr_readdata <= in_base;
                8'd6:  csr_readdata <= in_wr_seq;
                8'd7:  csr_readdata <= in_rd_seq;
                8'd8:  csr_readdata <= done_seq;
                8'd9:  csr_readdata <= {24'd0, theta};
                8'd10: csr_readdata <= CLOCK_HZ;
                8'd11: csr_readdata <= scratch;
                8'd12: csr_readdata <= T;
                8'd13: csr_readdata <= step_last;
                8'd14: csr_readdata <= lat_last;
                8'd16: csr_readdata <= {28'd0, s_pred};
                8'd17: csr_readdata <= {16'd0, s_windows};
                8'd18: csr_readdata <= {24'd0, s_wcount};
                8'd19: csr_readdata <= {12'd0, s_spk1};
                8'd20: csr_readdata <= {12'd0, s_spk2};
                8'd21: csr_readdata <= s_events;
                8'd24: csr_readdata <= {{4{s_win[0][27]}}, s_win[0]};
                8'd25: csr_readdata <= {{4{s_win[1][27]}}, s_win[1]};
                8'd26: csr_readdata <= {{4{s_win[2][27]}}, s_win[2]};
                8'd27: csr_readdata <= {{4{s_win[3][27]}}, s_win[3]};
                8'd28: csr_readdata <= {{4{s_win[4][27]}}, s_win[4]};
                8'd29: csr_readdata <= {{4{s_win[5][27]}}, s_win[5]};
                8'd32: csr_readdata <= {{4{s_cur[0][27]}}, s_cur[0]};
                8'd33: csr_readdata <= {{4{s_cur[1][27]}}, s_cur[1]};
                8'd34: csr_readdata <= {{4{s_cur[2][27]}}, s_cur[2]};
                8'd35: csr_readdata <= {{4{s_cur[3][27]}}, s_cur[3]};
                8'd36: csr_readdata <= {{4{s_cur[4][27]}}, s_cur[4]};
                8'd37: csr_readdata <= {{4{s_cur[5][27]}}, s_cur[5]};
                8'd40: csr_readdata <= {{4{s_score[0][27]}}, s_score[0]};
                8'd41: csr_readdata <= {{4{s_score[1][27]}}, s_score[1]};
                8'd42: csr_readdata <= {{4{s_score[2][27]}}, s_score[2]};
                8'd43: csr_readdata <= {{4{s_score[3][27]}}, s_score[3]};
                8'd44: csr_readdata <= {{4{s_score[4][27]}}, s_score[4]};
                8'd45: csr_readdata <= {{4{s_score[5][27]}}, s_score[5]};
                default: csr_readdata <= 32'd0;
            endcase
        end
    end

    assign st_pred    = s_pred;
    assign st_wcount  = s_wcount;
    assign st_ready   = ctrl_ready;
    assign st_enabled = enable;
    assign st_error   = st_overrun | st_emu_err;
    assign st_done    = done_seq;
endmodule
