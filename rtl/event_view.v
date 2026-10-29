`timescale 1ns/1ps
// event_view.v -- VGA event view (roadmap 4.11): ON events green, OFF events red, no event dark grey,
// 128x128 shown 3x (384x384) centred in 640x480.
//
// Double-buffered so the picture never tears: events of the current window are written to bank W,
// the VGA scan reads bank D. At the first line of vertical blanking, if the emulator is idle and at
// least `win` frames have completed since the last swap, the banks swap and the new W bank is
// cleared (16384 clocks, well inside the ~36,000-clock blanking interval). While clearing,
// ev_ready is low (the event FIFO absorbs, and the emulator back-pressures its source).
// An event writes 2'b01 (ON) or 2'b10 (OFF); a later event on the same pixel in the same window wins.
//
// Latency from the VGA timing inputs (x, y, de, hs_n, vs_n) to the outputs is 3 clocks; the
// synchronisation outputs are delayed to match.
module event_view (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        ev_valid,
    output wire        ev_ready,
    input  wire [31:0] ev_data,
    input  wire        emu_idle,
    input  wire [31:0] frame_cnt,
    input  wire [1:0]  cfg_win,        // window = 1, 2, 4 or 8 emulator frames

    input  wire [10:0] x,
    input  wire [10:0] y,
    input  wire        de,
    input  wire        hs_n,
    input  wire        vs_n,
    input  wire        vblank,

    output reg  [7:0]  r,
    output reg  [7:0]  g,
    output reg  [7:0]  b,
    output reg         de_o,
    output reg         hs_o,
    output reg         vs_o,

    output reg  [15:0] disp_first,     // frames shown on screen: first .. last (first > last: none yet)
    output reg  [15:0] disp_last
);
    localparam [10:0] X0 = 11'd128;
    localparam [10:0] Y0 = 11'd48;
    localparam [10:0] SZ = 11'd384;

    // ---------------- write side ----------------
    reg         wbank, dbank;
    reg         clearing;
    reg [14:0]  clr_cnt;
    reg         init_sweep;            // after reset: zero both banks
    reg [31:0]  win_base;
    reg         vblank_d;

    wire [3:0]  win         = 4'd1 << cfg_win;
    wire        vb_rise     = vblank && !vblank_d;
    wire        window_done = emu_idle && ((frame_cnt - win_base) >= {28'd0, win});
    wire        swap        = vb_rise && window_done && !clearing && !init_sweep;

    assign ev_ready = !clearing && !init_sweep;

    wire        ev_wr    = ev_valid && ev_ready;
    wire        ev_pol   = ev_data[31];
    wire [6:0]  ev_y     = ev_data[30:24];
    wire [6:0]  ev_x     = ev_data[23:17];
    wire        ram_we    = ev_wr || clearing || init_sweep;
    wire [14:0] ram_waddr = init_sweep ? clr_cnt :
                            clearing   ? {wbank, clr_cnt[13:0]} : {wbank, ev_y, ev_x};
    wire [1:0]  ram_wdata = (clearing || init_sweep) ? 2'b00 : (ev_pol ? 2'b01 : 2'b10);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wbank <= 1'b0; dbank <= 1'b1; clearing <= 1'b0; clr_cnt <= 15'd0; init_sweep <= 1'b1;
            win_base <= 32'd0; vblank_d <= 1'b0; disp_first <= 16'd1; disp_last <= 16'd0;
        end else begin
            vblank_d <= vblank;
            if (init_sweep) begin
                clr_cnt <= clr_cnt + 15'd1;
                if (clr_cnt == 15'h7FFF) begin init_sweep <= 1'b0; clr_cnt <= 15'd0; end
            end else if (clearing) begin
                clr_cnt <= clr_cnt + 15'd1;
                if (clr_cnt[13:0] == 14'h3FFF) begin clearing <= 1'b0; clr_cnt <= 15'd0; end
            end else if (swap) begin
                dbank      <= wbank;
                wbank      <= ~wbank;
                disp_first <= win_base[15:0];
                disp_last  <= frame_cnt[15:0] - 16'd1;
                win_base   <= frame_cnt;
                clearing   <= 1'b1;
                clr_cnt    <= 15'd0;
            end
        end
    end

    // ---------------- read side (VGA) ----------------
    wire [10:0] xr = x - X0;
    wire [10:0] yr = y - Y0;
    wire        in_panel = de && (x >= X0) && (x < X0 + SZ) && (y >= Y0) && (y < Y0 + SZ);
    // floor(n / 3) for 0 <= n < 384
    wire [18:0] xm = xr * 8'd171;
    wire [18:0] ym = yr * 8'd171;

    reg         a_panel, a_de, a_hs, a_vs;
    reg [6:0]   a_xi, a_yi;
    reg         b_panel, b_de, b_hs, b_vs;
    wire [1:0]  q;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_panel <= 1'b0; a_de <= 1'b0; a_hs <= 1'b1; a_vs <= 1'b1; a_xi <= 7'd0; a_yi <= 7'd0;
            b_panel <= 1'b0; b_de <= 1'b0; b_hs <= 1'b1; b_vs <= 1'b1;
            r <= 8'd0; g <= 8'd0; b <= 8'd0; de_o <= 1'b0; hs_o <= 1'b1; vs_o <= 1'b1;
        end else begin
            a_panel <= in_panel; a_de <= de; a_hs <= hs_n; a_vs <= vs_n;
            a_xi <= xm[15:9]; a_yi <= ym[15:9];
            b_panel <= a_panel; b_de <= a_de; b_hs <= a_hs; b_vs <= a_vs;
            de_o <= b_de; hs_o <= b_hs; vs_o <= b_vs;
            if (!b_de)            begin r <= 8'd0;   g <= 8'd0;   b <= 8'd0;   end
            else if (!b_panel)    begin r <= 8'd0;   g <= 8'd0;   b <= 8'd0;   end
            else if (q[0])        begin r <= 8'd0;   g <= 8'd255; b <= 8'd0;   end
            else if (q[1])        begin r <= 8'd255; g <= 8'd0;   b <= 8'd0;   end
            else                  begin r <= 8'd28;  g <= 8'd28;  b <= 8'd28;  end
        end
    end

    sdp_ram #(.DW(2), .AW(15)) u_disp (
        .clk(clk),
        .we(ram_we), .waddr(ram_waddr), .wdata(ram_wdata),
        .re(1'b1),   .raddr({dbank, a_yi, a_xi}), .rdata(q)
    );
endmodule
